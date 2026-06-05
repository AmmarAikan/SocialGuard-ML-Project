from pathlib import Path
import json
import pickle

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

from train_models import (
    BASE_DIR,
    DATA_DIR,
    MODEL_DIR,
    REPORT_DIR,
    build_user_level_dataset,
    create_features_and_target,
    slugify_model_name,
)

app = Flask(__name__)

MODEL_CACHE = {}
ASSETS = {}


def _load_pickle_or_joblib(path_joblib: Path, path_pkl: Path = None):
    if path_joblib.exists():
        return joblib.load(path_joblib)
    if path_pkl and path_pkl.exists():
        with open(path_pkl, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"Missing artifact: {path_joblib}")


def load_assets(force: bool = False):
    global ASSETS
    if ASSETS and not force:
        return ASSETS

    metadata_path = MODEL_DIR / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    preprocessor = _load_pickle_or_joblib(MODEL_DIR / "preprocessor.joblib")
    pca = _load_pickle_or_joblib(MODEL_DIR / "pca.joblib")
    feature_names = _load_pickle_or_joblib(MODEL_DIR / "feature_names.joblib", BASE_DIR / "models_pkl" / "feature_names.pkl")
    categorical_values = _load_pickle_or_joblib(MODEL_DIR / "categorical_values.joblib", BASE_DIR / "models_pkl" / "categorical_values.pkl")
    default_values = _load_pickle_or_joblib(MODEL_DIR / "default_values.joblib", BASE_DIR / "models_pkl" / "default_values.pkl")

    df = build_user_level_dataset(save_processed=False)
    X, y = create_features_and_target(df)
    comparison = pd.read_csv(REPORT_DIR / "model_comparison.csv")

    ASSETS = {
        "metadata": metadata,
        "preprocessor": preprocessor,
        "pca": pca,
        "feature_names": feature_names,
        "categorical_values": categorical_values,
        "default_values": default_values,
        "df": df,
        "X": X,
        "y": y,
        "comparison": comparison,
    }
    return ASSETS


def available_model_names():
    metadata = load_assets()["metadata"]
    return list(metadata.get("all_models", {}).keys())


def get_model(model_name: str):
    if model_name not in MODEL_CACHE:
        path = MODEL_DIR / f"{slugify_model_name(model_name)}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_name}")
        MODEL_CACHE[model_name] = joblib.load(path)
    return MODEL_CACHE[model_name]


def _to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _to_binary(value):
    if isinstance(value, str):
        return 1 if value.lower() in ["1", "true", "yes", "on"] else 0
    return int(bool(value))


def prepare_manual_input(features: dict) -> pd.DataFrame:
    assets = load_assets()
    feature_names = assets["feature_names"]
    defaults = dict(assets["default_values"])

    row = {col: defaults.get(col, 0) for col in feature_names}

    # Numerical values exposed in the web form.
    numerical_inputs = [item["name"] for item in assets["metadata"]["manual_form_config"]["numeric_inputs"]]
    for name in numerical_inputs:
        if name in features and name in row:
            row[name] = _to_float(features.get(name), defaults.get(name, 0))

    # Extra raw value used for deriving posting_intensity.
    account_age_days = _to_float(features.get("account_age_days"), defaults.get("account_age_days", 365))

    # Boolean profile indicators.
    for col in assets["metadata"]["manual_form_config"]["boolean_inputs"]:
        if col in row:
            row[col] = _to_binary(features.get(col, row.get(col, 0)))

    # Categorical dropdowns.
    for col in assets["metadata"]["manual_form_config"]["categorical_inputs"]:
        if col in row and features.get(col) not in [None, ""]:
            row[col] = str(features.get(col))

    # Derived features so manual mode behaves like the training pipeline.
    post_count = max(_to_float(row.get("post_count"), 1), 1)
    likes_mean = _to_float(row.get("likes_mean"), 0)
    comments_mean = _to_float(row.get("comments_mean"), 0)
    shares_mean = _to_float(row.get("shares_mean"), 0)
    total_engagement = (likes_mean + comments_mean + shares_mean) * post_count

    if "likes_sum" in row:
        row["likes_sum"] = likes_mean * post_count
    if "comments_sum" in row:
        row["comments_sum"] = comments_mean * post_count
    if "shares_sum" in row:
        row["shares_sum"] = shares_mean * post_count
    if "likes_max" in row:
        row["likes_max"] = max(_to_float(row.get("likes_max"), 0), likes_mean)
    if "comments_max" in row:
        row["comments_max"] = max(_to_float(row.get("comments_max"), 0), comments_mean)
    if "shares_max" in row:
        row["shares_max"] = max(_to_float(row.get("shares_max"), 0), shares_mean)
    if "hashtag_sum" in row:
        row["hashtag_sum"] = _to_float(row.get("hashtag_mean"), 0) * post_count
    if "weekend_activity_rate" in row:
        row["weekend_activity_rate"] = _to_float(row.get("weekend_rate"), 0)
    if "engagement_rate" in row:
        row["engagement_rate"] = total_engagement / (_to_float(row.get("followers_count"), 0) + 1)
    if "avg_reaction_per_post" in row:
        row["avg_reaction_per_post"] = total_engagement / (post_count + 1)
    if "followers_following_ratio" in row:
        row["followers_following_ratio"] = _to_float(row.get("followers_count"), 0) / (_to_float(row.get("following_count"), 0) + 1)
    if "posting_intensity" in row:
        row["posting_intensity"] = _to_float(row.get("posts_count"), 0) / (account_age_days + 1)
    if "profile_signal_score" in row:
        signal_cols = ["profile_picture", "profile_banner", "has_bio", "has_website", "has_location", "is_verified"]
        row["profile_signal_score"] = sum(_to_binary(row.get(c, 0)) for c in signal_cols)

    input_df = pd.DataFrame([row])[feature_names]
    return input_df.replace([np.inf, -np.inf], np.nan)


def predict_from_dataframe(input_df: pd.DataFrame, model_name: str) -> dict:
    assets = load_assets()
    model = get_model(model_name)
    transformed = assets["preprocessor"].transform(input_df)
    reduced = assets["pca"].transform(transformed)
    prediction_int = int(model.predict(reduced)[0])
    predicted = "Fake" if prediction_int == 1 else "Real"

    fake_probability = None
    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(reduced)[0]
        fake_probability = float(probs[1]) if len(probs) > 1 else None
        confidence = float(np.max(probs))

    if fake_probability is None:
        risk = "Medium" if predicted == "Fake" else "Low"
    elif fake_probability >= 0.70:
        risk = "High"
    elif fake_probability >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "predicted": predicted,
        "prediction_int": prediction_int,
        "fake_probability": None if fake_probability is None else round(fake_probability * 100, 2),
        "confidence": None if confidence is None else round(confidence * 100, 2),
        "risk": risk,
        "model": model_name,
    }


def build_summary(input_df: pd.DataFrame) -> dict:
    row = input_df.iloc[0]
    keys = [
        "profile_completeness", "followers_count", "following_count", "posts_count",
        "post_count", "likes_mean", "comments_mean", "shares_mean",
        "engagement_rate", "followers_following_ratio", "profile_signal_score", "dominant_platform"
    ]
    summary = {}
    for key in keys:
        if key in row.index:
            value = row[key]
            if isinstance(value, (float, np.floating)):
                value = round(float(value), 3)
            elif isinstance(value, (int, np.integer)):
                value = int(value)
            elif isinstance(value, (bool, np.bool_)):
                value = bool(value)
            else:
                value = str(value)
            summary[key] = value
    return summary


@app.route("/")
def index():
    assets = load_assets()
    metadata = assets["metadata"]
    config = metadata["manual_form_config"]
    defaults = assets["default_values"]
    comparison_rows = assets["comparison"].round(3).to_dict(orient="records")
    return render_template(
        "index.html",
        metadata=metadata,
        model_names=available_model_names(),
        selected_model=metadata.get("best_model", "SVM"),
        comparison_rows=comparison_rows,
        total_rows=len(assets["df"]),
        numeric_inputs=config["numeric_inputs"],
        boolean_inputs=config["boolean_inputs"],
        categorical_inputs=config["categorical_inputs"],
        categorical_values=assets["categorical_values"],
        defaults=defaults,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        assets = load_assets()
        model_name = data.get("model_name") or assets["metadata"].get("best_model", "SVM")
        if model_name not in available_model_names():
            return jsonify({"success": False, "error": "Invalid model selected."}), 400

        input_mode = data.get("input_mode", "row")
        actual = None
        user_index = None
        if input_mode == "row":
            user_index = int(data.get("user_index", 0))
            if user_index < 0 or user_index >= len(assets["X"]):
                return jsonify({"success": False, "error": f"Row index must be between 0 and {len(assets['X']) - 1}."}), 400
            input_df = assets["X"].iloc[[user_index]].copy()
            actual = "Fake" if int(assets["y"].iloc[user_index]) == 1 else "Real"
        else:
            input_df = prepare_manual_input(data.get("features", {}))

        result = predict_from_dataframe(input_df, model_name)
        result.update({
            "success": True,
            "input_mode": input_mode,
            "actual": actual,
            "user_index": user_index,
            "is_correct": None if actual is None else (actual == result["predicted"]),
            "summary": build_summary(input_df),
        })
        return jsonify(result)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/reports/<path:filename>")
def report_file(filename):
    return send_from_directory(REPORT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

"""
Fake Social Media Profile Detection - Training Script
Cybersecurity Machine Learning Project

This script converts the two Kaggle CSV files into one user-level dataset,
cleans the data, applies preprocessing, handles class imbalance with SMOTE,
performs feature engineering, trains the six required classification models,
checks overfitting, and saves deployment artifacts for the Flask web app/API.
"""

import json
import pickle
import shutil
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PKL_DIR = BASE_DIR / "models_pkl"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PKL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# These columns are intentionally excluded because they are high-cardinality metadata,
# identifiers, PII, or direct proxy variables that can make evaluation unrealistically
# easy and raise overfitting/data-leakage concerns.
LEAKAGE_CONTROL_DROP_COLUMNS = [
    "home_country",
    "language_preference",
    "dominant_language",
    "dominant_media_type",
    "char_mean",
    "char_max",
    "account_age_days",
    "url_rate",
    "url_to_post_ratio",
]

IMPORTANT_NUMERIC_FEATURES = [
    "profile_completeness",
    "followers_count",
    "following_count",
    "posts_count",
    "post_count",
    "likes_mean",
    "comments_mean",
    "shares_mean",
    "engagement_rate",
    "followers_following_ratio",
    "posting_intensity",
    "profile_signal_score",
    "weekend_activity_rate",
    "avg_reaction_per_post",
]

MANUAL_NUMERIC_INPUTS = [
    {"name": "profile_completeness", "label": "Profile Completeness", "min": 0, "max": 1, "step": 0.01},
    {"name": "account_age_days", "label": "Account Age (Days)", "min": 1, "max": 5000, "step": 1},
    {"name": "followers_count", "label": "Followers Count", "min": 0, "max": 1000000, "step": 1},
    {"name": "following_count", "label": "Following Count", "min": 0, "max": 1000000, "step": 1},
    {"name": "posts_count", "label": "Profile Posts Count", "min": 0, "max": 100000, "step": 1},
    {"name": "post_count", "label": "Analyzed Activity Count", "min": 1, "max": 10000, "step": 1},
    {"name": "likes_mean", "label": "Average Likes", "min": 0, "max": 100000, "step": 0.1},
    {"name": "comments_mean", "label": "Average Comments", "min": 0, "max": 100000, "step": 0.1},
    {"name": "shares_mean", "label": "Average Shares", "min": 0, "max": 100000, "step": 0.1},
    {"name": "hour_mean", "label": "Average Posting Hour", "min": 0, "max": 23, "step": 0.1},
    {"name": "weekend_rate", "label": "Weekend Activity Rate", "min": 0, "max": 1, "step": 0.01},
    {"name": "media_rate", "label": "Media Usage Rate", "min": 0, "max": 1, "step": 0.01},
    {"name": "hashtag_mean", "label": "Average Hashtags", "min": 0, "max": 100, "step": 0.1},
]

MANUAL_BOOLEAN_INPUTS = [
    "is_private",
    "is_verified",
    "profile_picture",
    "profile_banner",
    "has_bio",
    "has_website",
    "has_location",
]

MANUAL_CATEGORICAL_INPUTS = ["dominant_device", "dominant_platform"]


def safe_one_hot_encoder():
    """Create OneHotEncoder compatible with both old and new sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn fallback
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def mode_or_unknown(series: pd.Series) -> str:
    mode_values = series.mode(dropna=True)
    return mode_values.iloc[0] if len(mode_values) else "unknown"


def safe_float(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def build_user_level_dataset(save_processed: bool = True) -> pd.DataFrame:
    """Read the two raw CSV files, aggregate activity data by user_id, and merge."""
    profiles_path = DATA_DIR / "raw_user_profiles.csv"
    activities_path = DATA_DIR / "raw_user_activities.csv"

    profiles = pd.read_csv(profiles_path)
    activities = pd.read_csv(activities_path)

    # Cleaning: remove duplicates, parse dates, standardize booleans.
    profiles = profiles.drop_duplicates(subset=["user_id"]).copy()
    activities = activities.drop_duplicates(subset=["activity_id"]).copy()
    activities["timestamp"] = pd.to_datetime(activities["timestamp"], errors="coerce")

    bool_cols_profiles = [
        "is_private", "is_verified", "profile_picture", "profile_banner",
        "has_bio", "has_website", "has_location", "is_fake"
    ]
    bool_cols_activities = ["is_weekend", "has_media", "is_fake"]
    for col in bool_cols_profiles:
        if col in profiles.columns:
            profiles[col] = profiles[col].astype(bool)
    for col in bool_cols_activities:
        if col in activities.columns:
            activities[col] = activities[col].astype(bool)

    # Aggregate post/activity behavior per user instead of training on duplicated posts.
    activity_features = activities.groupby("user_id").agg(
        post_count=("activity_id", "count"),
        likes_mean=("likes", "mean"), likes_sum=("likes", "sum"), likes_max=("likes", "max"),
        comments_mean=("comments", "mean"), comments_sum=("comments", "sum"), comments_max=("comments", "max"),
        shares_mean=("shares", "mean"), shares_sum=("shares", "sum"), shares_max=("shares", "max"),
        hour_mean=("hour_of_day", "mean"), hour_std=("hour_of_day", "std"),
        day_mean=("day_of_week", "mean"), day_std=("day_of_week", "std"),
        weekend_rate=("is_weekend", "mean"), media_rate=("has_media", "mean"),
        char_mean=("character_count", "mean"), char_max=("character_count", "max"),
        hashtag_mean=("hashtag_count", "mean"), hashtag_sum=("hashtag_count", "sum"),
        mention_mean=("mention_count", "mean"), mention_sum=("mention_count", "sum"),
        url_rate=("contains_url", "mean"),
    ).reset_index()

    for col in ["device", "platform", "media_type", "language"]:
        if col in activities.columns:
            mode_df = activities.groupby("user_id")[col].agg(mode_or_unknown).reset_index(name=f"dominant_{col}")
            activity_features = activity_features.merge(mode_df, on="user_id", how="left")

    df = profiles.merge(activity_features, on="user_id", how="left")

    # Feature Engineering Process 1: cybersecurity/domain behavior features.
    df["engagement_rate"] = (df["likes_sum"] + df["comments_sum"] + df["shares_sum"]) / (df["followers_count"] + 1)
    df["followers_following_ratio"] = df["followers_count"] / (df["following_count"] + 1)
    df["posting_intensity"] = df["posts_count"] / (df["account_age_days"] + 1)
    profile_cols = ["profile_picture", "profile_banner", "has_bio", "has_website", "has_location", "is_verified"]
    df["profile_signal_score"] = df[profile_cols].astype(int).sum(axis=1)
    df["url_to_post_ratio"] = df["url_rate"].fillna(0)
    df["weekend_activity_rate"] = df["weekend_rate"].fillna(0)
    df["avg_reaction_per_post"] = (df["likes_sum"] + df["comments_sum"] + df["shares_sum"]) / (df["post_count"] + 1)

    df = df.replace([np.inf, -np.inf], np.nan)
    if save_processed:
        df.to_csv(DATA_DIR / "processed_user_dataset.csv", index=False)
    return df


def create_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Create X and y while removing identifiers, PII, target, and leakage-prone columns."""
    y = df["is_fake"].astype(int)

    columns_to_drop = [
        "is_fake", "user_id", "username", "full_name", "first_name", "last_name",
        "email", "creation_date", "home_city", "home_region", "account_type",
        *LEAKAGE_CONTROL_DROP_COLUMNS,
    ]
    X = df.drop(columns=[c for c in columns_to_drop if c in df.columns]).copy()

    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)

    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", safe_one_hot_encoder()),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    return preprocessor, numeric_features, categorical_features


def get_models() -> Dict[str, object]:
    """Six required classification algorithms with regularization to reduce overfitting."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=8, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=0.7, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=13),
        "SVM": SVC(
            kernel="rbf", C=0.8, gamma="scale", probability=True,
            random_state=RANDOM_STATE
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=70, learning_rate=0.7, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=80, max_depth=2, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=4,
            tree_method="hist", eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=1
        ),
    }


def slugify_model_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def build_artifact_dictionaries(X: pd.DataFrame, numeric_features: list, categorical_features: list) -> Tuple[dict, dict, dict]:
    """Prepare dropdown values, defaults, and manual input metadata for deployment."""
    categorical_values = {}
    for col in categorical_features:
        vals = X[col].dropna().astype(str).sort_values().unique().tolist()
        categorical_values[col] = vals if vals else ["unknown"]

    default_values = {}
    for col in X.columns:
        if col in numeric_features:
            default_values[col] = safe_float(X[col].median(), 0)
        else:
            default_values[col] = mode_or_unknown(X[col].astype(str))

    # Extra values used to compute engineered manual-input fields.
    df_processed = pd.read_csv(DATA_DIR / "processed_user_dataset.csv") if (DATA_DIR / "processed_user_dataset.csv").exists() else pd.DataFrame()
    if "account_age_days" in df_processed.columns:
        default_values["account_age_days"] = safe_float(df_processed["account_age_days"].median(), 365)
    else:
        default_values["account_age_days"] = 365.0

    manual_form_config = {
        "numeric_inputs": MANUAL_NUMERIC_INPUTS,
        "boolean_inputs": MANUAL_BOOLEAN_INPUTS,
        "categorical_inputs": MANUAL_CATEGORICAL_INPUTS,
        "default_values": default_values,
    }
    return categorical_values, default_values, manual_form_config


def get_transformed_feature_names(preprocessor, numeric_features: list, categorical_features: list) -> list:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        names = list(numeric_features)
        try:
            encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_names = encoder.get_feature_names_out(categorical_features).tolist()
            names.extend(cat_names)
        except Exception:
            names.extend(categorical_features)
        return names


def save_pickle_and_joblib(obj, name: str):
    joblib.dump(obj, MODEL_DIR / f"{name}.joblib")
    with open(MODEL_PKL_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def generate_eda_report_plots(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    """Save explanatory plots used in the notebook and web dashboard."""
    # 1. Class distribution.
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["Real", "Fake"], counts.values)
    plt.title("Target Distribution: Real vs Fake")
    plt.ylabel("Number of User Profiles")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "eda_class_distribution.png", dpi=160)
    plt.close()

    # 2. Missing values.
    missing = df.isna().sum().sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 4))
    plt.bar(missing.index, missing.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top Missing Values After Merging the Two CSV Files")
    plt.ylabel("Missing Count")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "eda_missing_values.png", dpi=160)
    plt.close()

    # 3. Important feature distributions.
    features = [f for f in ["followers_count", "following_count", "posts_count", "post_count", "likes_mean", "comments_mean", "shares_mean", "engagement_rate"] if f in df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()
    for ax, col in zip(axes, features):
        values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(values, bins=25)
        ax.set_title(col)
    for ax in axes[len(features):]:
        ax.axis("off")
    fig.suptitle("Distributions of Key Profile and Activity Features", fontsize=14)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "eda_key_feature_distributions.png", dpi=160)
    plt.close(fig)

    # 4. Boxplots for important features by class.
    box_cols = [f for f in ["profile_completeness", "followers_count", "following_count", "engagement_rate", "avg_reaction_per_post"] if f in df.columns]
    fig, axes = plt.subplots(1, len(box_cols), figsize=(4 * len(box_cols), 4))
    if len(box_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, box_cols):
        real = df.loc[df["is_fake"] == False, col].replace([np.inf, -np.inf], np.nan).dropna()
        fake = df.loc[df["is_fake"] == True, col].replace([np.inf, -np.inf], np.nan).dropna()
        ax.boxplot([real, fake], labels=["Real", "Fake"], showfliers=False)
        ax.set_title(col)
    fig.suptitle("Important Features by Target Class", fontsize=14)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "eda_key_feature_boxplots.png", dpi=160)
    plt.close(fig)

    # 5. Correlation heatmap.
    numeric_for_corr = X.select_dtypes(include="number")
    corr = numeric_for_corr.corr().fillna(0)
    selected_cols = corr.abs().mean().sort_values(ascending=False).head(12).index
    small_corr = numeric_for_corr[selected_cols].corr().fillna(0)
    plt.figure(figsize=(9, 7))
    plt.imshow(small_corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(selected_cols)), selected_cols, rotation=60, ha="right")
    plt.yticks(range(len(selected_cols)), selected_cols)
    plt.title("Correlation Heatmap for Important Numeric Features")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "eda_correlation_heatmap.png", dpi=160)
    plt.close()

    # 6. Target correlation for numeric features.
    target_corr = numeric_for_corr.apply(lambda col: col.corr(y)).abs().sort_values(ascending=False).head(14)
    plt.figure(figsize=(10, 4))
    plt.bar(target_corr.index, target_corr.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top Numeric Features Related to the Target")
    plt.ylabel("Absolute Correlation")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "eda_target_relationship.png", dpi=160)
    plt.close()


def train_and_evaluate() -> pd.DataFrame:
    print("Loading and preparing dataset...")
    df = build_user_level_dataset(save_processed=True)
    X, y = create_features_and_target(df)
    generate_eda_report_plots(df, X, y)

    print(f"Records: {df.shape[0]} | Raw columns after merge: {df.shape[1]} | Features used: {X.shape[1]}")
    print("Target distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    transformed_feature_names = get_transformed_feature_names(preprocessor, numeric_features, categorical_features)

    # Feature Engineering Process 2: PCA dimensionality reduction.
    pca_components = min(18, X_train_processed.shape[1])
    pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_processed)
    X_test_pca = pca.transform(X_test_processed)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.title("PCA Cumulative Explained Variance")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "pca_explained_variance.png", dpi=160)
    plt.close()

    # Imbalanced data handling using SMOTE on the training set only.
    before_counts = pd.Series(y_train).value_counts().sort_index()
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_pca, y_train)
    after_counts = pd.Series(y_train_balanced).value_counts().sort_index()

    balance_df = pd.DataFrame({
        "Before SMOTE": before_counts,
        "After SMOTE": after_counts,
    }).fillna(0)
    balance_df.index = ["Real", "Fake"]
    balance_df.plot(kind="bar", figsize=(7, 4))
    plt.title("Class Balance Before and After SMOTE")
    plt.ylabel("Training Records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "smote_balance.png", dpi=160)
    plt.close()

    models = get_models()
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_balanced, y_train_balanced)

        y_train_pred = model.predict(X_train_pca)
        y_test_pred = model.predict(X_test_pca)

        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
        results.append({
            "Model": name,
            "Train Accuracy": accuracy_score(y_train, y_train_pred),
            "Test Accuracy": accuracy_score(y_test, y_test_pred),
            "Precision": precision_score(y_test, y_test_pred, zero_division=0),
            "Recall": recall_score(y_test, y_test_pred, zero_division=0),
            "Train F1-Score": train_f1,
            "Test F1-Score": test_f1,
            "Generalization Gap": abs(train_f1 - test_f1),
        })
        trained_models[name] = model

        report = classification_report(y_test, y_test_pred, target_names=["Real", "Fake"], zero_division=0)
        (REPORT_DIR / f"{slugify_model_name(name)}_classification_report.txt").write_text(report, encoding="utf-8")

    results_df = pd.DataFrame(results).sort_values(by="Test F1-Score", ascending=False)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    results_df[["Model", "Train Accuracy", "Test Accuracy", "Train F1-Score", "Test F1-Score", "Generalization Gap"]].to_csv(
        REPORT_DIR / "overfitting_check.csv", index=False
    )
    print("\nModel comparison:")
    print(results_df)

    best_model_name = str(results_df.iloc[0]["Model"])
    best_model = trained_models[best_model_name]

    # Save all models for the Flask demo dropdown in both joblib and pkl formats.
    all_model_paths = {}
    for name, model in trained_models.items():
        slug = slugify_model_name(name)
        joblib.dump(model, MODEL_DIR / f"{slug}.joblib")
        with open(MODEL_PKL_DIR / f"{slug}.pkl", "wb") as f:
            pickle.dump(model, f)
        all_model_paths[name] = {"joblib": f"models/{slug}.joblib", "pkl": f"models_pkl/{slug}.pkl"}

    # Save full inference artifacts and separate metadata objects like the example project.
    categorical_values, default_values, manual_form_config = build_artifact_dictionaries(X, numeric_features, categorical_features)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(["Real", "Fake"], dtype=object)

    save_pickle_and_joblib(preprocessor, "preprocessor")
    save_pickle_and_joblib(pca, "pca")
    save_pickle_and_joblib(preprocessor.named_transformers_["num"].named_steps["scaler"], "standard_scaler")
    save_pickle_and_joblib(label_encoder, "label_encoder")
    save_pickle_and_joblib(X.columns.tolist(), "feature_names")
    save_pickle_and_joblib(transformed_feature_names, "transformed_feature_names")
    save_pickle_and_joblib(categorical_values, "categorical_values")
    save_pickle_and_joblib(default_values, "default_values")

    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    with open(MODEL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_PKL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    y_best_pred = best_model.predict(X_test_pca)
    ConfusionMatrixDisplay.from_predictions(y_test, y_best_pred, display_labels=["Real", "Fake"])
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix_best_model.png", dpi=160)
    plt.close()

    # Comparison visualizations.
    plt.figure(figsize=(10, 5))
    plt.bar(results_df["Model"], results_df["Test F1-Score"])
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Model Comparison by Test F1-Score")
    plt.ylabel("Test F1-Score")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison_f1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(results_df["Model"], results_df["Test Accuracy"])
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Model Comparison by Test Accuracy")
    plt.ylabel("Test Accuracy")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(results_df["Model"], results_df["Generalization Gap"])
    plt.xticks(rotation=30, ha="right")
    plt.title("Overfitting Check: Train/Test F1 Gap")
    plt.ylabel("Absolute F1 Gap")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "overfitting_gap.png", dpi=160)
    plt.close()

    metadata = {
        "project_name": "SocialGuard AI - Fake Social Media Profile Detection",
        "project_scope": "Detect fake social media accounts from profile information and aggregated activity behavior to support cybersecurity, fraud prevention, and social-engineering risk reduction.",
        "best_model": best_model_name,
        "all_models": all_model_paths,
        "features_used": X.columns.tolist(),
        "transformed_feature_names": transformed_feature_names,
        "removed_for_leakage_control": LEAKAGE_CONTROL_DROP_COLUMNS,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "important_numeric_features": IMPORTANT_NUMERIC_FEATURES,
        "manual_form_config": manual_form_config,
        "target_mapping": {"0": "Real", "1": "Fake"},
        "pca_components": int(pca_components),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "records": int(df.shape[0]),
        "raw_columns_after_merge": int(df.shape[1]),
        "features_before_encoding": int(X.shape[1]),
        "train_records": int(X_train.shape[0]),
        "test_records": int(X_test.shape[0]),
        "smote_records_after_resampling": int(len(y_train_balanced)),
        "test_accuracy_best": float(results_df.iloc[0]["Test Accuracy"]),
        "test_f1_best": float(results_df.iloc[0]["Test F1-Score"]),
        "best_generalization_gap": float(results_df.iloc[0]["Generalization Gap"]),
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (MODEL_DIR / "preprocessing_metadata.json").write_text(json.dumps({
        "feature_names": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categorical_values": categorical_values,
        "default_values": default_values,
        "manual_form_config": manual_form_config,
    }, indent=2), encoding="utf-8")

    print(f"\nBest model: {best_model_name}")
    print("Saved: models/best_model.joblib and models/best_model.pkl")
    print("Saved API/UI assets: feature_names, categorical_values, default_values, standard_scaler, preprocessor, PCA")
    print("Saved overfitting check: reports/overfitting_check.csv")
    return results_df


if __name__ == "__main__":
    train_and_evaluate()

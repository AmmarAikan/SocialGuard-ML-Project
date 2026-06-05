from pathlib import Path
import json
import pickle

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from train_models import (
    BASE_DIR,
    MODEL_DIR,
    REPORT_DIR,
    build_user_level_dataset,
    create_features_and_target,
    slugify_model_name,
)
from app import prepare_manual_input, predict_from_dataframe, available_model_names, load_assets

api = Flask(__name__)


@api.route('/api/status', methods=['GET'])
def status():
    assets = load_assets()
    metadata = assets['metadata']
    return jsonify({
        'status': 'online',
        'project': metadata.get('project_name'),
        'best_model': metadata.get('best_model'),
        'available_algorithms': available_model_names(),
        'records': metadata.get('records'),
        'features_required': {
            'manual_numeric_inputs': [x['name'] for x in metadata['manual_form_config']['numeric_inputs']],
            'manual_boolean_inputs': metadata['manual_form_config']['boolean_inputs'],
            'manual_categorical_inputs': metadata['manual_form_config']['categorical_inputs'],
        }
    })


@api.route('/api/metadata', methods=['GET'])
def metadata():
    assets = load_assets()
    return jsonify({
        'categorical_values': assets['categorical_values'],
        'default_values': assets['default_values'],
        'model_metadata': assets['metadata'],
    })


@api.route('/api/sample-row/<int:index>', methods=['GET'])
def sample_row(index):
    assets = load_assets()
    if index < 0 or index >= len(assets['X']):
        return jsonify({'success': False, 'error': f'Index must be between 0 and {len(assets["X"]) - 1}'}), 400
    row = assets['X'].iloc[index].replace([np.inf, -np.inf], np.nan).fillna('').to_dict()
    actual = 'Fake' if int(assets['y'].iloc[index]) == 1 else 'Real'
    return jsonify({'success': True, 'index': index, 'actual': actual, 'features': row})


@api.route('/api/predict', methods=['POST'])
def predict_api():
    """
    Predict a user profile as Real/Fake.

    Option 1 - dataset row mode:
    {
      "algorithm": "SVM",
      "mode": "row",
      "row_index": 31
    }

    Option 2 - manual mode:
    {
      "algorithm": "SVM",
      "mode": "manual",
      "features": {
        "profile_completeness": 0.72,
        "account_age_days": 300,
        "followers_count": 120,
        "following_count": 900,
        "posts_count": 12,
        "post_count": 20,
        "likes_mean": 5,
        "comments_mean": 1,
        "shares_mean": 0,
        "is_private": 0,
        "is_verified": 0,
        "profile_picture": 1,
        "profile_banner": 0,
        "has_bio": 1,
        "has_website": 0,
        "has_location": 0,
        "dominant_device": "Android",
        "dominant_platform": "Mobile App"
      }
    }
    """
    try:
        data = request.get_json(force=True)
        assets = load_assets()
        algorithm = data.get('algorithm') or data.get('model_name') or assets['metadata'].get('best_model')
        if algorithm not in available_model_names():
            return jsonify({'success': False, 'error': f'Invalid algorithm. Choose from {available_model_names()}'}), 400

        mode = data.get('mode', data.get('input_mode', 'manual'))
        actual = None
        if mode == 'row':
            row_index = int(data.get('row_index', data.get('user_index', 0)))
            if row_index < 0 or row_index >= len(assets['X']):
                return jsonify({'success': False, 'error': f'row_index must be between 0 and {len(assets["X"]) - 1}'}), 400
            input_df = assets['X'].iloc[[row_index]].copy()
            actual = 'Fake' if int(assets['y'].iloc[row_index]) == 1 else 'Real'
        else:
            input_df = prepare_manual_input(data.get('features', {}))
            row_index = None

        result = predict_from_dataframe(input_df, algorithm)
        result.update({'success': True, 'mode': mode, 'row_index': row_index, 'actual': actual})
        return jsonify(result)
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


if __name__ == '__main__':
    api.run(host='0.0.0.0', port=5001, debug=True)

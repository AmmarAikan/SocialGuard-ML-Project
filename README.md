# SocialGuard AI: Fake Social Media Profile Detection Using Machine Learning

## Project Scope

### Business Problem
Fake social media profiles are frequently used in cybersecurity attacks such as phishing, impersonation, spam campaigns, fake engagement, misinformation, and social engineering. Organizations need an automated way to identify suspicious accounts before they are used to harm users or damage platform trust.

### Project Objective
This project builds a **binary classification machine learning system** that predicts whether a social media account is **Real** or **Fake** using profile information and aggregated user activity behavior.

### In Scope
- Merge two Kaggle CSV files into one user-level dataset.
- Detect fake/real profiles using six machine learning algorithms.
- Perform exploratory data analysis, data cleaning, preprocessing, feature engineering, model training, evaluation, and deployment.
- Provide both a Flask web dashboard and a REST API.
- Allow prediction by either selecting a dataset row index or manually entering user/profile values.

### Out of Scope
- Real-time connection to an actual social media platform.
- Collecting private user data from live accounts.
- Multi-class classification of fake account types. This project focuses on binary detection: Real vs Fake.

---

## Dataset
The project uses the Kaggle dataset **Fake Profile and Post Detection on Social Media**. The raw dataset contains two CSV files:

- `data/raw_user_profiles.csv`: account/profile-level data.
- `data/raw_user_activities.csv`: post/activity-level data.

The two-file issue is solved by aggregating post/activity records by `user_id`, then merging the aggregated features with profile records. This creates one clean user-level dataset: `data/processed_user_dataset.csv`.

---

## Methodology

### 1. Data Discovery and Cleaning
- Loaded both raw CSV files.
- Removed duplicate users and duplicate activities.
- Parsed timestamp values.
- Standardized Boolean columns.
- Merged profile and activity data at user level.
- Removed identifiers and PII such as names, usernames, emails, and user IDs.
- Controlled potential data leakage by excluding overly strong direct-proxy columns.
- Handled missing values using `SimpleImputer` inside the preprocessing pipeline.

### 2. Exploratory Data Analysis
The notebook includes more than 8 EDA analyses/plots, including:
- Class distribution.
- Missing-value analysis.
- Key feature distributions.
- Boxplots for important features.
- Correlation heatmap.
- Engagement behavior by class.
- Dominant platform distribution by class.
- IQR outlier detection.
- Target relationship analysis.

### 3. Preprocessing and Feature Engineering
- **Encoding:** Categorical features are converted to numerical values using One-Hot Encoding.
- **Scaling:** Numerical features are standardized using StandardScaler.
- **Missing Values:** Median imputation for numerical features and most-frequent imputation for categorical features.
- **Imbalanced Data:** SMOTE is applied only on the training set.
- **Feature Engineering Process 1:** Domain-specific features such as engagement rate, follower/following ratio, posting intensity, profile signal score, weekend activity rate, and average reaction per post.
- **Feature Engineering Process 2:** PCA dimensionality reduction after preprocessing.

### 4. Model Training
The project implements all six required classification models:

1. Random Forest
2. Logistic Regression
3. K-Nearest Neighbors (KNN)
4. Support Vector Machine (SVM)
5. AdaBoost
6. XGBoost

### 5. Evaluation and Deployment
- Model comparison table is saved in `reports/model_comparison.csv`.
- Overfitting/generalization check is saved in `reports/overfitting_check.csv`.
- The final best model is saved in both `.joblib` and `.pkl` formats.
- The web dashboard supports both dataset-row prediction and manual-value prediction.
- A REST API is provided through `api_app.py`.

---

## Current Results

The best model is **SVM** with approximately:

- Test Accuracy: 0.977
- Test F1-Score: 0.955
- Generalization Gap: 0.009

The results meet the required minimum of 80% Accuracy and F1-Score while keeping a small train/test gap, reducing the previous concern about unrealistic 100% performance.

---

## Saved Artifacts

The project saves deployment artifacts similar to the example project:

```text
models/
├── best_model.joblib
├── best_model.pkl
├── preprocessor.joblib
├── pca.joblib
├── standard_scaler.joblib
├── label_encoder.joblib
├── feature_names.joblib
├── transformed_feature_names.joblib
├── categorical_values.joblib
├── default_values.joblib
├── model_metadata.json
└── preprocessing_metadata.json

models_pkl/
├── best_model.pkl
├── preprocessor.pkl
├── pca.pkl
├── standard_scaler.pkl
├── label_encoder.pkl
├── feature_names.pkl
├── transformed_feature_names.pkl
├── categorical_values.pkl
├── default_values.pkl
└── one .pkl file for each trained model
```

---

## How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Retrain Models and Regenerate Reports
```bash
python train_models.py
```

### 3. Open the Notebook
```bash
jupyter notebook final_machine_learning_project.ipynb
```

The first code cell includes `%matplotlib inline` so plots display inside Jupyter Notebook.

### 4. Run the Web Dashboard
```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

### 5. Run the REST API
```bash
python api_app.py
```

API server:

```text
http://127.0.0.1:5001
```

Available endpoints:

```text
GET  /api/status
GET  /api/metadata
GET  /api/sample-row/<index>
POST /api/predict
```

---

## Example API Request

```json
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
```

---

## Project Structure

```text
Fake_Profile_Detection_Project/
├── data/
│   ├── raw_user_profiles.csv
│   ├── raw_user_activities.csv
│   └── processed_user_dataset.csv
├── models/
├── models_pkl/
├── reports/
├── templates/
│   └── index.html
├── app.py
├── api_app.py
├── train_models.py
├── final_machine_learning_project.ipynb
├── requirements.txt
├── SocialGuard_API.postman_collection.json
├── SUBMISSION_CHECKLIST.md
├── run_training.bat / run_training.sh
├── run_app.bat / run_app.sh
├── run_api.bat / run_api.sh
└── README.md
```

---

## Team

- Student 1: Ammar Aikan
- Student 2: Yousif Farouk
- Student 3: Aiban Mohammed
- Student 4:Naser Ali



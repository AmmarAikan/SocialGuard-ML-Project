# Submission Checklist

Before submitting, confirm the following:

## Pass/Fail Constraints
- [x] Dataset has more than 1,000 records.
- [x] Dataset has more than 10 features.
- [x] Accuracy and F1-Score are above 80%.
- [ ] Team size is exactly 2 or 3 students. Do not submit 4 names.

## Technical Requirements
- [x] 8+ EDA plots/analyses in the notebook.
- [x] Missing values handled using imputers.
- [x] Duplicate/noisy records handled.
- [x] Categorical encoding using One-Hot Encoding.
- [x] Numerical scaling using StandardScaler.
- [x] Imbalanced data handled with SMOTE on training data only.
- [x] Feature Engineering 1: domain-specific features.
- [x] Feature Engineering 2: PCA.
- [x] Six required classification models trained.
- [x] Comparison report generated.
- [x] Overfitting/generalization gap report generated.
- [x] Best model saved as `.joblib` and `.pkl`.
- [x] Web dashboard included.
- [x] REST API included.

## Final Manual Steps
- [ ] Open `README.md` and make sure the team names are correct.
- [ ] Run `python train_models.py` without errors.
- [ ] Run `python app.py` and test row-index prediction.
- [ ] Test manual-value prediction in the web page.
- [ ] Zip the full project folder, not only the notebook.

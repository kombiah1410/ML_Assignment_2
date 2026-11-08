# Handwritten Digit Recognition | Machine Learning Assignment 2

## a. Problem statement
Classify handwritten numerals (0 through 9) from grayscale 8×8 pixel images using five classical machine-learning classifiers. Compare six metrics on the **same held-out test set**, and offer a Streamlit interface to evaluate the **saved** models on uploaded CSV files without retraining.

## b. Dataset description
- **Dataset:** Optical Recognition of Handwritten Digits, 8×8 version, available via `sklearn.datasets.load_digits` (derived from the UCI Optical Recognition of Handwritten Digits dataset).
- **Original dataset documentation:** https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
- **UCI repository:** https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- **Observations:** 1,797; **features:** 64 numeric pixel-intensity values (`pixel_0_0` through `pixel_7_7`); **target:** `digit` (10 classes, 0–9).
- **Split:** stratified 80% training (1,437) and 20% testing (360), random seed 1907.
- **Target labels:** 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. Pixel intensities are 0–16.
- **Preprocessing:** median imputation and standard scaling are fitted on training data only, saved inside each pipeline.
- **AUC:** multiclass one-vs-rest, support-weighted; precision, recall and F1: support-weighted; MCC: multiclass MCC.
- **Dataset file:** `data/handwritten_digits.csv`; held-out file: `test_data.csv` with 64 features and `digit`.

## c. GitHub repository link
**To be added after you publish this revised project:** `<YOUR_NEW_GITHUB_REPOSITORY_LINK>`.

## Live Streamlit app link
**To be added after deploying the revised repository:** `<YOUR_NEW_STREAMLIT_APP_LINK>`.

## d. Models used and evaluation comparison
The following are actual metrics computed by `train_models.py` on the saved 360-row test split:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9694 | 0.9994 | 0.9706 | 0.9694 | 0.9695 | 0.9662 |
| Decision Tree | 0.8583 | 0.9213 | 0.8651 | 0.8583 | 0.8586 | 0.8432 |
| KNN | 0.9861 | 0.9983 | 0.9866 | 0.9861 | 0.9860 | 0.9846 |
| Naive Bayes | 0.8167 | 0.9875 | 0.8524 | 0.8167 | 0.8144 | 0.8007 |
| Support Vector Machine | 0.9889 | 0.9966 | 0.9893 | 0.9889 | 0.9889 | 0.9877 |


## Performance observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | 96.94% test accuracy, 0.9994 AUC (highest AUC); strong performance, but train accuracy 99.86% suggests a modest generalization gap. |
| Decision Tree | 85.83% test accuracy compared with 100% train accuracy, showing pronounced overfitting for the unrestricted tree. |
| KNN | 98.61% test accuracy and 0.9846 MCC; local similarity works very well for the 8x8 pixel representation. |
| Naive Bayes | 81.67% accuracy and 0.8144 F1; its conditional-independence assumption is unrealistic for neighboring pixels. |
| Support Vector Machine | 98.89% accuracy and 0.9877 MCC, the highest of all five models on this held-out test split. |
| **Overall winner** | **Support Vector Machine**, by test accuracy (0.9889), weighted F1 (0.9889), and MCC (0.9877). Logistic Regression leads on AUC alone. |

## Reproduce and test
Run these commands in the BITS Virtual Lab (do not use a local run as a substitute for the required lab screenshot):

```bash
python -m pip install -r requirements.txt
python train_models.py --data data/handwritten_digits.csv --target digit --seed 1907
streamlit run app.py
```

Upload `test_data.csv` (or check the built-in test-data checkbox), select a classifier, and inspect the metrics and confusion matrix. The app loads serialized fitted pipelines and **never fits models on uploaded test data**.

## Repository contents
```
app.py
train_models.py
metrics_utils.py
requirements.txt
README.md
data/handwritten_digits.csv
test_data.csv
model/metadata.json
model/metrics.csv
model/metrics_table.md
model/train_vs_test.csv
model/*.joblib  (five saved fitted pipelines)
.streamlit/config.toml
```

## Execution evidence and deployment
- BITS Virtual Lab screenshot: **not included**; capture a genuine screenshot after running the commands above in the lab.
- GitHub repository and Streamlit Cloud: **not published by this package**. Add your real links after completing those steps.
- Results depend on the fixed train/test split and package versions in `requirements.txt`.

## Academic integrity
Use this project as a working implementation to inspect, adapt and explain in your own submission. The assignment requires individual work and specifically restricts direct copy-paste AI submissions.

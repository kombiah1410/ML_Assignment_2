"""
Trains the 5 required classifiers on ONE dataset, evaluates them on a held-out
test split, and saves everything the Streamlit app needs.

Usage (run this on BITS Virtual Lab and take your screenshot while it runs):
    python train_models.py --data data/<your_dataset>.csv --target <TargetColumn>
Optional:
    --drop id,customerID      columns that are identifiers / leak the target
    --test-size 0.2           fraction held out as test_data.csv
    --seed 7                  pick YOUR OWN seed, don't keep a default
    --tune                    small GridSearchCV per model (5-fold, train split only)

Outputs:
    model/<model>.joblib      full pipeline = preprocessing + classifier
    model/metadata.json       feature list, class labels, versions
    model/metrics.csv         test-set metrics (goes in README table)
    model/metrics_table.md    same table, already in README markdown format
    model/train_vs_test.csv   train vs test accuracy (use it for observations)
    test_data.csv             held-out rows, with target column, for app upload
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from metrics_utils import METRIC_NAMES, accuracy_score, averaging_for, compute_metrics

MIN_FEATURES, MIN_ROWS = 12, 500

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    # dense output: GaussianNB cannot take sparse matrices
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def model_specs(seed):
    """(estimator, small param grid used only with --tune)"""
    return {
        "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=seed),
                                {"clf__C": [0.1, 1, 10]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=seed),
                          {"clf__max_depth": [None, 5, 10, 20], "clf__min_samples_leaf": [1, 5, 20]}),
        "KNN": (KNeighborsClassifier(n_neighbors=5),
                {"clf__n_neighbors": [3, 5, 11, 21], "clf__weights": ["uniform", "distance"]}),
        "Naive Bayes": (GaussianNB(), {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]}),
        # AUC needs probabilities. SVC(probability=True) is deprecated from sklearn 1.9,
        # so the SVC is wrapped in 5-fold Platt calibration (same idea). Slow on >20k rows.
        "Support Vector Machine": (CalibratedClassifierCV(SVC(kernel="rbf", random_state=seed),
                                                          method="sigmoid", cv=5, ensemble=False),
                                   {"clf__estimator__C": [0.5, 1, 5]}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--drop", default="")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--positive", default=None,
                    help="binary only: the label that counts as 'positive' for Precision/Recall/F1")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df.columns = [c.strip() for c in df.columns]
    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not found. Columns: {list(df.columns)}")

    drop = [c.strip() for c in args.drop.split(",") if c.strip()]
    missing_drop = [c for c in drop if c not in df.columns]
    if missing_drop:
        raise SystemExit(f"--drop columns not found: {missing_drop}")
    df = df.drop(columns=drop)
    df = df.dropna(subset=[args.target]).drop_duplicates().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != args.target]
    # Assignment hard requirements - fail loudly instead of losing marks silently
    if len(feature_cols) < MIN_FEATURES:
        raise SystemExit(f"Only {len(feature_cols)} features; assignment needs >= {MIN_FEATURES}.")
    if len(df) < MIN_ROWS:
        raise SystemExit(f"Only {len(df)} rows after cleaning; assignment needs >= {MIN_ROWS}.")

    X = df[feature_cols].copy()
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]
    for c in cat_cols:
        X[c] = X[c].where(X[c].isna(), X[c].astype(str))  # keep NaN for the imputer

    le = LabelEncoder()
    y_str = df[args.target].astype(str)
    le.fit(y_str)
    if args.positive is not None:
        if len(le.classes_) != 2 or args.positive not in le.classes_:
            raise SystemExit(f"--positive must be one of the 2 labels {list(le.classes_)}")
        neg = [c for c in le.classes_ if c != args.positive][0]
        le.classes_ = np.array([neg, args.positive])  # positive -> index 1
    y = le.transform(y_str)
    n_classes = len(le.classes_)
    if n_classes < 2:
        raise SystemExit("Target has a single class - not a classification problem.")
    print(f"Rows={len(df)}  features={len(feature_cols)} (num={len(num_cols)}, cat={len(cat_cols)})  "
          f"classes={n_classes}: {[str(c) for c in le.classes_]}")
    print("Class distribution:", dict(zip(le.classes_, np.bincount(y))))

    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=args.test_size, stratify=y, random_state=args.seed)
    X_train, X_test, y_train, y_test = X.iloc[idx_train], X.iloc[idx_test], y[idx_train], y[idx_test]

    # Held-out test file for the Streamlit app (original values, includes target)
    df.iloc[idx_test].to_csv("test_data.csv", index=False)

    os.makedirs("model", exist_ok=True)
    rows, overfit_rows = [], []
    for name, (est, grid) in model_specs(args.seed).items():
        t0 = time.time()
        pipe = Pipeline([("prep", build_preprocessor(num_cols, cat_cols)), ("clf", est)])
        if args.tune:
            search = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=args.seed),
                                  scoring="f1_weighted", n_jobs=-1)
            search.fit(X_train, y_train)
            pipe, best = search.best_estimator_, search.best_params_
        else:
            pipe.fit(X_train, y_train)
            best = {}
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)
        m = compute_metrics(y_test, y_pred, y_proba, n_classes)
        train_acc = accuracy_score(y_train, pipe.predict(X_train))
        joblib.dump(pipe, os.path.join("model", MODEL_FILES[name]), compress=3)
        rows.append({"ML Model Name": name, **m})
        overfit_rows.append({"ML Model Name": name, "Train Accuracy": train_acc,
                             "Test Accuracy": m["Accuracy"], "Gap": train_acc - m["Accuracy"],
                             "Best Params": json.dumps(best), "Fit seconds": round(time.time() - t0, 1)})
        print(f"{name:24s} " + "  ".join(f"{k}={v:.4f}" for k, v in m.items())
              + f"  (train acc {train_acc:.4f}, {time.time() - t0:.1f}s)")

    res = pd.DataFrame(rows)
    res.to_csv("model/metrics.csv", index=False)
    pd.DataFrame(overfit_rows).to_csv("model/train_vs_test.csv", index=False)

    lines = ["| ML Model Name | " + " | ".join(METRIC_NAMES) + " |",
             "|---|" + "---|" * len(METRIC_NAMES)]
    for r in rows:
        lines.append(f"| {r['ML Model Name']} | " + " | ".join(f"{r[k]:.4f}" for k in METRIC_NAMES) + " |")
    with open("model/metrics_table.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    meta = {
        "target": args.target,
        "feature_columns": feature_cols,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "classes": [str(c) for c in le.classes_],
        "averaging": averaging_for(n_classes),
        "positive_class": str(le.classes_[1]) if n_classes == 2 else None,
        "models": MODEL_FILES,
        "n_train": int(len(idx_train)), "n_test": int(len(idx_test)),
        "seed": args.seed, "tuned": args.tune,
        "dropped_columns": drop,
        "sklearn_version": sklearn.__version__,
    }
    with open("model/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nREADME table (model/metrics_table.md):\n" + "\n".join(lines))
    print(f"\nscikit-learn {sklearn.__version__} - pin this exact version in requirements.txt")


if __name__ == "__main__":
    main()

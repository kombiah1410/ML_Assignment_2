"""
Streamlit app: upload a TEST csv -> pick a saved model -> see metrics,
confusion matrix and classification report. Models are only LOADED and used
for prediction here; nothing is retrained on the uploaded data.
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from metrics_utils import METRIC_NAMES, compute_metrics, confusion_df, report_df

DATASET_NAME = "Optical Recognition of Handwritten Digits (UCI / scikit-learn)"
DATASET_BLURB = ("Each row represents an 8×8 handwritten digit image using 64 pixel-intensity "
                 "features; the target is the digit from 0 to 9.")

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
SAMPLE_TEST = os.path.join(BASE, "test_data.csv")

st.set_page_config(page_title="Handwritten digit classifier lab", page_icon="🔢", layout="wide")


@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    models = {name: joblib.load(os.path.join(MODEL_DIR, fname))
              for name, fname in meta["models"].items()}
    return meta, models


def prepare(df, meta):
    """Validate the upload and return (X, y_encoded) or raise ValueError."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    target = meta["target"]
    if target not in df.columns:
        raise ValueError(f"The file has no target column '{target}'. "
                         "Upload the test CSV including the label column.")
    missing = [c for c in meta["feature_columns"] if c not in df.columns]
    if missing:
        raise ValueError(f"{len(missing)} required feature column(s) are missing: {missing}")
    df = df.dropna(subset=[target])
    if df.empty:
        raise ValueError("No rows with a target value were found.")
    X = df[meta["feature_columns"]].copy()
    for c in meta["categorical_columns"]:
        X[c] = X[c].where(X[c].isna(), X[c].astype(str))  # keep NaN for the imputer
    for c in meta["numeric_columns"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")  # bad cells -> NaN -> imputed
    labels = df[target].astype(str)
    # numeric labels like 1.0 vs 1 - normalise to the training spelling
    known = set(meta["classes"])
    labels = labels.map(lambda v: v if v in known else
                        (str(int(float(v))) if v.replace(".", "", 1).lstrip("-").isdigit()
                         and float(v).is_integer() and str(int(float(v))) in known else v))
    unknown = sorted(set(labels) - known)
    if unknown:
        raise ValueError(f"Target values not seen during training: {unknown}. "
                         f"Expected one of {meta['classes']}.")
    class_index = {c: i for i, c in enumerate(meta["classes"])}
    return X, labels.map(class_index).to_numpy()


def evaluate(model, X, y, n_classes):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return compute_metrics(y, y_pred, y_proba, n_classes), y_pred


def fmt(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


# ---------------------------------------------------------------- layout
try:
    meta, models = load_artifacts()
except FileNotFoundError:
    st.error("Saved models not found in model/. Run train_models.py and commit the model folder.")
    st.stop()

classes = meta["classes"]
st.title("Handwritten digit classifier lab")
st.write(f"**{DATASET_NAME}** — {DATASET_BLURB}")
st.caption(f"Target: `{meta['target']}` · {len(meta['feature_columns'])} features · "
           f"{len(classes)} classes · models trained on {meta['n_train']} rows, "
           f"never refit in this app.")

with st.sidebar:
    st.header("1. Test data")
    upload = st.file_uploader("Upload test CSV (features + target column)", type=["csv"])
    if os.path.exists(SAMPLE_TEST):
        with open(SAMPLE_TEST, "rb") as f:
            st.download_button("Download the held-out test_data.csv", f,
                               file_name="test_data.csv", mime="text/csv")
        use_sample = st.checkbox("Use the repository's test_data.csv instead", value=False)
    else:
        use_sample = False
    st.header("2. Model")
    model_name = st.selectbox("Choose a classifier", list(models.keys()), key="model_choice")
    avg_note = ("positive class = '" + str(meta["positive_class"]) + "'"
                if meta["averaging"] == "binary" else "weighted average over classes")
    st.caption(f"Precision / Recall / F1 use {avg_note}. AUC is "
               f"{'binary ROC-AUC' if len(classes) == 2 else 'one-vs-rest, weighted'}.")

if upload is not None:
    raw = pd.read_csv(upload)
    source = upload.name
elif use_sample:
    raw = pd.read_csv(SAMPLE_TEST)
    source = "test_data.csv (repository)"
else:
    st.info("Upload a test CSV in the sidebar to evaluate the saved models. "
            "Need one? Download the held-out test_data.csv there first.")
    st.stop()

try:
    X, y = prepare(raw, meta)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.success(f"Evaluating on {len(y)} rows from {source}.")
tab_one, tab_all, tab_data = st.tabs([f"{model_name}", "Compare all five models", "Uploaded data"])

with tab_one:
    metrics, y_pred = evaluate(models[model_name], X, y, len(classes))
    cols = st.columns(len(METRIC_NAMES))
    for col, k in zip(cols, METRIC_NAMES):
        col.metric(k, fmt(metrics[k]))
    if np.isnan(metrics["AUC"]):
        st.caption("AUC is undefined because the uploaded file contains only one class.")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_df(y, y_pred, classes)
        size = max(4, 0.7 * len(classes) + 2)
        fig, ax = plt.subplots(figsize=(size, size * 0.8))
        sns.heatmap(cm.values, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Classification report")
        st.dataframe(report_df(y, y_pred, classes).style.format("{:.3f}"),
                     width="stretch")

with tab_all:
    table = []
    for name, mdl in models.items():
        m, _ = evaluate(mdl, X, y, len(classes))
        table.append({"ML Model Name": name, **m})
    comp = pd.DataFrame(table).set_index("ML Model Name")
    st.dataframe(comp.style.format("{:.4f}", na_rep="n/a").highlight_max(axis=0, color="#cfe3f5"),
                 width="stretch")
    pick = st.selectbox("Metric to chart", METRIC_NAMES, index=METRIC_NAMES.index("F1"))
    st.bar_chart(comp[pick])
    st.download_button("Download this comparison as CSV", comp.to_csv().encode(),
                       file_name="model_comparison.csv", mime="text/csv")

with tab_data:
    st.write(f"{raw.shape[0]} rows × {raw.shape[1]} columns")
    st.dataframe(raw.head(200), width="stretch")
    st.write("Class counts in this file:")
    st.dataframe(pd.Series(y).map(dict(enumerate(classes))).value_counts().rename("rows"))

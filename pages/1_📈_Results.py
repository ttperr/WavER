import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

import import_data
from model.utils import load_data

DATA_FOLDER = import_data.DATA_FOLDER
THRESHOLD = 0.65

st.set_page_config(page_title="Results", page_icon="📈")

st.title("Results")

st.header("What dataset would you like to evaluate?")

dataset_name = st.selectbox("Select a dataset", list(import_data.DATASET_SUFFIX.keys()) + ["custom_dataset"])

output = st.file_uploader("Upload the CSV result file", type=["csv"])
df = None

if output is not None:
    df = pd.read_csv(output)

if st.button("Start Evaluation"):
    dataset_name_confirmed = dataset_name

    if df is not None:
        table_a_serialized, table_b_serialized, table_a, table_b, X_train, y_train, X_valid, y_valid, X_test_ids, y_test = load_data(
            os.path.join(DATA_FOLDER, dataset_name), remove_col_names=False, order_cols=False, return_tables=True)
        
        st.success("Dataset loaded successfully.")

        # Print roc_auc, accuracy, precision, recall, and f1
        st.header("Evaluation Metrics")
        accuracy, precision, recall, f1, roc_auc = None, None, None, None, None

        logits = df["Score"].values
        y_pred = [1 if logit > THRESHOLD else 0 for logit in logits]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, logits)

        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1: {f1}")
        st.write(f"ROC AUC: {roc_auc}")

        st.header("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig = plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

        st.header("ROC Curve")

        fpr, tpr, _ = roc_curve(y_test, logits)
        roc_auc = auc(fpr, tpr)

        fig = plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(fig)

        st.header("Examples of Predictions")

        # Separate indices for matches (1) and non-matches (0)
        match_indices = [i for i in range(len(y_pred)) if y_pred[i] == 1]
        non_match_indices = [i for i in range(len(y_pred)) if y_pred[i] == 0]

        # Select up to 3 examples from matches and 2 from non-matches (or vice versa if not enough matches)
        examples = match_indices[:3] + non_match_indices[:2]
        if len(examples) < 5:
            examples += match_indices[3:5 - len(examples)] + non_match_indices[2:5 - len(examples)]

        # Ensure exactly 5 examples, padded if necessary
        examples = examples[:5]

        for idx, example_idx in enumerate(examples):
            with st.expander(f"Example {idx + 1}"):
                st.markdown(f"### Prediction: **{'Matched' if y_pred[example_idx] == 1 else 'Not Matched'}**")
                st.markdown(f"**Actual:** {'Matched' if y_test[example_idx] == 1 else 'Not Matched'}")
                st.write("### Entity A")
                st.write(table_a.loc[X_test_ids[example_idx][0]])
                st.write("### Entity B")
                st.write(table_b.loc[X_test_ids[example_idx][1]])

    else:
        st.error("Please upload a CSV file to start the evaluation.")
else:
    st.error("Click the button above to start the evaluation.")
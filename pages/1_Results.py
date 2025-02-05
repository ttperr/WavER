import os
import re
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

import src.import_data as import_data
from model.utils import load_data

######## Constants ########

DATA_FOLDER = import_data.DATA_FOLDER
dataset_suffix_keys = set(import_data.DATASET_SUFFIX.keys())
regex = r"^results_([^_]+)\.csv$"
THRESHOLD = 0.65


######## Functions ########

def validate_filename(filename):
    match = re.match(regex, filename)
    if match:
        dataset_name = match.group(1)
        return dataset_name if dataset_name in dataset_suffix_keys or dataset_name == "custom_dataset" else None
    return None


def prepare_dataframe(data):
    df = pd.DataFrame(data)
    for col in df.columns:
        df[col] = df[col].astype(str)  # Convert all columns to strings
    return df


######## Page ########

# Initialize session state variables
if "results_loaded" not in st.session_state:
    st.session_state.results_loaded = False
    st.session_state.data = {}
    st.session_state.dataset_name = None
    st.session_state.uploaded_file_name = None

if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = None

st.set_page_config(page_title="Results", page_icon="📈", layout="wide")

st.sidebar.title("Presentation")
st.sidebar.subheader("About this page")
st.sidebar.write(
    "This page allows you to evaluate the results of the entity matching model.\n\n"
    "Upload a CSV file containing the results of the model, and the page will display the evaluation metrics, "
    "confusion matrix, ROC curve, and examples of predictions."
)

st.title("Results")

st.header("What dataset would you like to evaluate?")

# Track the uploaded file name
uploaded_file = st.file_uploader("Upload the CSV result file", type=["csv"])

# Reset session state if a new file is uploaded
if uploaded_file is not None:
    if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.results_loaded = False
        st.session_state.data = {}
        st.session_state.dataset_name = None
        st.session_state.uploaded_file_name = uploaded_file.name

if uploaded_file is not None and not st.session_state.results_loaded:
    filename = os.path.basename(uploaded_file.name)
    dataset_name = validate_filename(filename)
    if dataset_name is None:
        st.error(
            "The uploaded file is not a valid result file. A valid result file should be named as 'results_DATASET-NAME.csv'.")
    else:
        df = pd.read_csv(uploaded_file)
        table_a_serialized, table_b_serialized, table_a, table_b, X_train, y_train, X_valid, y_valid, X_test_ids, y_test = load_data(
            os.path.join(DATA_FOLDER, dataset_name), remove_col_names=False, order_cols=False, return_tables=True
        )
        logits = df["Score"].values
        y_pred = [1 if logit > THRESHOLD else 0 for logit in logits]

        # Save to session state
        st.session_state.results_loaded = True
        st.session_state.data = {
            "df": df,
            "table_a": table_a,
            "table_b": table_b,
            "X_test_ids": X_test_ids,
            "y_test": y_test,
            "y_pred": y_pred,
        }
        st.success("Dataset loaded successfully.")

if (st.session_state.output_df is not None and not st.session_state.results_loaded) or st.button("Load last result"):
    dataset_name = st.session_state.dataset_name
    df = st.session_state.output_df

    if dataset_name is None:
        st.error("Error: No dataset name found in the session state.")
        st.stop()

    table_a_serialized, table_b_serialized, table_a, table_b, X_train, y_train, X_valid, y_valid, X_test_ids, y_test = load_data(
        os.path.join(DATA_FOLDER, dataset_name), remove_col_names=False, order_cols=False, return_tables=True
    )
    logits = df["Score"].values
    y_pred = [1 if logit > THRESHOLD else 0 for logit in logits]

    # Save to session state
    st.session_state.results_loaded = True
    st.session_state.data = {
        "df": df,
        "table_a": table_a,
        "table_b": table_b,
        "X_test_ids": X_test_ids,
        "y_test": y_test,
        "y_pred": y_pred,
    }
    st.session_state.dataset_name = dataset_name
    st.success("Last result loaded successfully.")

if st.session_state.results_loaded:
    data = st.session_state.data
    df = data["df"]
    table_a = data["table_a"]
    table_b = data["table_b"]
    X_test_ids = data["X_test_ids"]
    y_test = data["y_test"]
    y_pred = data["y_pred"]

    st.header("Evaluation Metrics")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, df["Score"].values)

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
    fpr, tpr, _ = roc_curve(y_test, df["Score"].values)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    st.pyplot(fig)

    st.header("Examples of Predictions")

    # Separate indices for matches (1) and non-matches (0)
    match_indices = [i for i in range(len(y_pred)) if y_pred[i] == 1]
    non_match_indices = [i for i in range(len(y_pred)) if y_pred[i] == 0]

    examples = match_indices[:3] + non_match_indices[:2]
    if len(examples) < 5:
        examples += match_indices[3:5 - len(examples)] + non_match_indices[2:5 - len(examples)]
    examples = examples[:5]

    example_options = {f"Example {idx + 1}": example_idx for idx, example_idx in enumerate(examples)}

    selected_example_label = st.selectbox("Select an example to display:", list(example_options.keys()))
    selected_example_idx = example_options[selected_example_label]

    # Display the selected example details
    st.markdown(f"### Prediction: **{'Matched' if y_pred[selected_example_idx] == 1 else 'Not Matched'}**")
    st.markdown(f"**Actual:** {'Matched' if y_test[selected_example_idx] == 1 else 'Not Matched'}")

    # Get the entities
    entity_a = table_a.loc[X_test_ids[selected_example_idx][0]].to_dict()
    entity_b = table_b.loc[X_test_ids[selected_example_idx][1]].to_dict()

    # Get common and unique features
    common_features = set(entity_a.keys()).intersection(entity_b.keys())
    unique_a = set(entity_a.keys()) - common_features
    unique_b = set(entity_b.keys()) - common_features

    # Display common features in a table
    if common_features:
        st.markdown("### Common Features")
        common_data = [
            {"Feature": f"{feature}", "Entity A": entity_a[feature], "Entity B": entity_b[feature]}
            for feature in common_features
        ]
        common_df = prepare_dataframe(common_data)
        st.dataframe(common_df, hide_index=True, width=1920)

    # Display unique features of Entity A in a table
    if unique_a:
        st.markdown("### Unique Features in Entity A")
        unique_a_data = [{"Feature": f"{feature}", "Value": entity_a[feature]} for feature in unique_a]
        unique_a_df = prepare_dataframe(unique_a_data)
        st.dataframe(unique_a_df, hide_index=True, width=1920)

    # Display unique features of Entity B in a table
    if unique_b:
        st.markdown("### Unique Features in Entity B")
        unique_b_data = [{"Feature": f"{feature}", "Value": entity_b[feature]} for feature in unique_b]
        unique_b_df = prepare_dataframe(unique_b_data)
        st.dataframe(unique_b_df, hide_index=True, width=1920)

import os
import pickle
import time

import streamlit as st

import import_data
from model.utils import load_data, save_tables
from model.Blocking import merge_true_matches, perform_blocking_sbert, get_blocking_metrics, perform_blocking_tfidf, merge_indices
from model.CrossEncoder import prepare_data_cross_encoder, fit_cross_encoder, evaluate_cross_encoder

######## Constants ########

DATA_DIR = import_data.DATA_FOLDER
EPOCHS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

######## Session state ########

if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = None

if 'known_pairs' not in st.session_state:
    st.session_state.known_pairs = False

if "blocking_metrics" not in st.session_state:
    st.session_state.blocking_metrics = {
        "reduction_ratio": None,
        "recall": None,
        "f1": None
    }

if 'blocked_pairs' not in st.session_state:
    st.session_state.blocked_pairs = None

if 'cross_encoder_metrics' not in st.session_state:
    st.session_state.cross_encoder_metrics = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None
    }

######## Streamlit app ########

st.title("Entity Resolution")
st.write(
    "Tristan PERROT degree project for KTH @ Wavestone")

######## Dataset ########

st.header("Dataset")

dataset_option = st.radio(
    "Select an option",
    ["Use a benchmark dataset", "Use a custom dataset"]
)

if dataset_option == "Use a benchmark dataset":
    dataset_name = st.selectbox(
        "Select the dataset",
        import_data.DATASET_SUFFIX.keys(), index=2
    )

    st.session_state.dataset_name = dataset_name
    st.session_state.known_pairs = True

    if st.button("Download dataset"):
        downloaded = import_data.download_dataset(dataset_name)
        if downloaded:
            st.write("Dataset downloaded successfully")
        else:
            st.write("Dataset already exists, skipping download. If you want to re-download, please delete the folder first.")
else:
    tableA = st.file_uploader("Upload the table A (format: subject_id, col1, col2,..)", type=['csv'])
    tableB = st.file_uploader("Upload the table B (format: subject_id, col1, col2,..)", type=['csv'])
    pairs = st.file_uploader("Upload the true matches (format: idA, idB) (optional)", type=['csv'])

    st.session_state.dataset_name = "custom_dataset"
    st.session_state.known_pairs = pairs is not None or "custom_dataset" in os.listdir(DATA_DIR)

    if st.button("Upload dataset"):
        if tableA is not None and tableB is not None:
            save_tables(tableA, tableB, DATA_DIR, pairs=pairs)
            st.write("Dataset uploaded successfully")
        else:
            st.warning("Please upload both tables")

######## Blocking ########

st.header("Model")
st.subheader("Blocking")

pair_used = st.radio(
    "Select the pairs to use",
    [ "Already included pairs", "Blocked pairs"] if st.session_state.known_pairs else ["Blocked pairs"]
)

if pair_used == "Blocked pairs":
    blocking = st.selectbox(
        "Select the blocking KNN method to use",
        [
            'sentence-transformers/allenai-specter',
            'all-distilroberta-v1',
            'all-mpnet-base-v2',
            'multi-qa-mpnet-base-dot-v1',
        ],
        index=3
    )

    columns = st.columns(3)
    with columns[0]:
        merge_with_tfidf = st.checkbox("Merge with TF-IDF (recommended)", key="merge_with_tfidf_blocking", value=True)
    with columns[1]:
        remove_col_name = st.checkbox("Remove column name (recommended)", key="remove_col_name_blocking", value=True)
    with columns[2]:
        order_cols = st.checkbox("Order columns (not recommended)", key="order_cols_blocking")

    n_neighbors = st.number_input("Number of neighbors to retrieve", value=15, min_value=1, max_value=40)

    if st.button("Run blocking"):
        with st.status("Blocking..."):
            st.write("Blocking started")
            
            custom_dataset = st.session_state.dataset_name == "custom_dataset"
            table_a_serialized, table_b_serialized, X_train_ids, y_train, X_valid_ids, y_valid, X_test_ids, y_test = load_data(os.path.join(DATA_DIR, st.session_state.dataset_name), remove_col_names=remove_col_name, order_cols=order_cols, custom_dataset=custom_dataset)
            all_true_matches = merge_true_matches(X_train_ids, y_train, X_valid_ids, y_valid, X_test_ids, y_test) if X_train_ids is not None else None
            
            st.write("True matches merged")
            st.write("Starting blocking")
            
            blocked_pairs = perform_blocking_sbert(blocking, table_a_serialized, table_b_serialized, n_neighbors=n_neighbors, metric='cosine', device='cpu')
            if merge_with_tfidf:
                blocked_pairs = merge_indices(blocked_pairs, perform_blocking_tfidf(table_a_serialized, table_b_serialized, n_neighbors=n_neighbors, metric='cosine'))
                st.write("Merged with TF-IDF")
            
            st.write("Blocking done")
            
            st.session_state.blocked_pairs = blocked_pairs
            
            reduction_ratio, recall, f1 = get_blocking_metrics(blocked_pairs, all_true_matches, len(table_a_serialized), len(table_b_serialized)) if all_true_matches is not None else (None, None, None)

            st.session_state.blocking_metrics["reduction_ratio"] = reduction_ratio
            st.session_state.blocking_metrics["recall"] = recall
            st.session_state.blocking_metrics["f1"] = f1

# Display metrics from session state
if st.session_state.blocking_metrics["reduction_ratio"] is not None:
    st.write(f"Reduction ratio: {st.session_state.blocking_metrics['reduction_ratio']}")
    st.write(f"Recall: {st.session_state.blocking_metrics['recall']}")
    st.write(f"F1 score: {st.session_state.blocking_metrics['f1']}")

######## Matching ########

st.subheader("Matching")

# Choose beetween supervised and unsupervised
matching = st.selectbox(
    "Select the matching method to use",
    ['supervised', 'unsupervised'] if st.session_state.known_pairs else ['unsupervised'],
)

columns = st.columns(2)
with columns[0]:
    remove_col_name = st.checkbox("Remove column name (recommended)", key="remove_col_name_matching", value=True)
with columns[1]:
    order_cols = st.checkbox("Order columns (recommended)", key="order_cols_matching", value=True)

if matching == 'supervised':
    model_name = st.selectbox(
        "Select the model to use",
        [
            'cross-encoder/stsb-roberta-base',
            'cross-encoder/stsb-distilroberta-base',
            'cross-encoder/ms-marco-MiniLM-L-12-v2',
        ],
        index=1
    )
    if st.button("Run matching"):
        with st.status("Matching in progress..."):
            st.write("Matching started")
            
            custom_dataset = st.session_state.dataset_name == "custom_dataset"
            train_loader, valid_set, y_valid, test_set, y_test = prepare_data_cross_encoder(os.path.join(DATA_DIR, st.session_state.dataset_name), remove_col_names=remove_col_name, order_cols=order_cols, custom_dataset=custom_dataset, blocked_pairs=st.session_state.blocked_pairs)
            
            st.write("Data prepared")
            
            logits, train_time = fit_cross_encoder(model_name, train_loader, valid_set, y_valid, test_set, epochs=EPOCHS, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            
            st.write("Model fitted")
            
            accuracy, precision, recall, f1, roc_auc = evaluate_cross_encoder(logits, y_test)
            
            st.write("Matching done")

            st.session_state.cross_encoder_metrics["accuracy"] = accuracy
            st.session_state.cross_encoder_metrics["precision"] = precision
            st.session_state.cross_encoder_metrics["recall"] = recall
            st.session_state.cross_encoder_metrics["f1"] = f1
            st.session_state.cross_encoder_metrics["roc_auc"] = roc_auc

        # Print an example of the predictions
        # Check the first five 1 predictions
        if st.session_state.cross_encoder_metrics["accuracy"] is not None:
            st.write("Example of predictions")
            for i in range(len(test_set)):
                if y_test[i] == 1:
                    st.write(f"Prediction: {logits[i]} - True value: {y_test[i]}")
                    st.write("Example:")
                    st.write(f"Entity A: {test_set[i][0]}")
                    st.write(f"Entity B: {test_set[i][1]}")
                    if i == 5:
                        break


    # Display metrics from session state
    if st.session_state.cross_encoder_metrics["accuracy"] is not None:
        st.write(f"Accuracy: {st.session_state.cross_encoder_metrics['accuracy']}")
        st.write(f"Precision: {st.session_state.cross_encoder_metrics['precision']}")
        st.write(f"Recall: {st.session_state.cross_encoder_metrics['recall']}")
        st.write(f"F1 score: {st.session_state.cross_encoder_metrics['f1']}")
        st.write(f"ROC AUC: {st.session_state.cross_encoder_metrics['roc_auc']}")

else:
    # Unsupervised
    unsupervised_method = st.selectbox(
        "Select the unsupervised method to use",
        ['ZeroShot Embedding', 'LLM Inference'],
        index=0
    )

    if st.button("Run matching"):
        if unsupervised_method == 'ZeroShot Embedding':
            model_name_zero_shot = st.selectbox(
                "Select the model to use",
                [
                    'all-mpnet-base-v2',
                    'sentence-transformers/multi-qa-mpnet-base-dot-v1',
                    'facebook/bart-large-mnli',
                ],
                index=1
            )

            with st.status("Matching in progress..."):
                # TODO: implement zero shot embedding
                pass

        st.write("Matching done")
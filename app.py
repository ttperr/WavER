import os

import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import import_data
from model.Blocking import (get_blocking_metrics, merge_indices,
                            merge_true_matches, perform_blocking_sbert,
                            perform_blocking_tfidf)
from model.CrossEncoder import (evaluate_cross_encoder, fit_cross_encoder,
                                prepare_data_cross_encoder)
from model.utils import load_data, save_tables
from model.ZeroShot import evaluate_zero_shot

######## Constants ########

DATA_FOLDER = import_data.DATA_FOLDER
EPOCHS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
THRESHOLD = 0.7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NOT_SPLIT_OPTION = "Not split matches"
ALREADY_SPLIT_OPTION = "Already split matches"
TESTING_PAIR_OPTION = "Testing pairs"
BLOCKED_PAIRS_OPTION = "Blocked pairs"
BENCHMARK_DATASET_OPTION = "Use a benchmark dataset"
CUSTOM_DATASET_OPTION = "Use a custom dataset"
NO_TRUE_MATCHES_OPTION = "I don't have any true matches"
CUSTOM_DATASET_NAME = "custom_dataset"

######## Session state ########

if 'supervised' not in st.session_state:
    st.session_state.supervised = True

if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = None

if 'known_pairs' not in st.session_state:
    st.session_state.known_pairs = False

if 'testing' not in st.session_state:
    st.session_state.testing = False

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

if 'unsupervied_metrics' not in st.session_state:
    st.session_state.unsupervised_metrics = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None
    }

######## Streamlit app ########

st.title("WavER - Entity Resolution Project")
st.write("Tristan PERROT degree project for KTH @ Wavestone")

st.session_state.supervised = st.selectbox(
    "Select the type of entity resolution method to use",
    ["Supervised", "Unsupervised"],
    index=0
) == "Supervised"

######## Dataset ########

st.header("Dataset")

dataset_option = st.radio(
    "Select an option",
    [BENCHMARK_DATASET_OPTION, CUSTOM_DATASET_OPTION]
)

if dataset_option == BENCHMARK_DATASET_OPTION:
    dataset_name = st.selectbox(
        "Select the dataset",
        import_data.DATASET_SUFFIX.keys(), index=2
    )

    st.session_state.dataset_name = dataset_name
    st.session_state.known_pairs = any(
        file.startswith("gs_") for file in os.listdir(os.path.join(DATA_FOLDER, dataset_name)))

    if st.button("Download dataset"):
        downloaded = import_data.download_dataset(dataset_name)
        if downloaded:
            st.write("Dataset downloaded successfully")
        else:
            st.write(
                "Dataset already exists, skipping download. If you want to re-download, please delete the folder first.")
else:
    tableA = st.file_uploader("Upload the table A (format: subject_id, col1, col2,..)", type=['csv'])
    tableB = st.file_uploader("Upload the table B (format: subject_id, col1, col2,..)", type=['csv'])

    train_pairs, val_pairs, test_pairs = None, None, None

    if st.session_state.supervised:
        splitted_pairs = st.selectbox(
            "Select the format of the your true matches",
            [NOT_SPLIT_OPTION, ALREADY_SPLIT_OPTION]
        )

        if splitted_pairs == NOT_SPLIT_OPTION:
            train_pairs = st.file_uploader("Upload the train pairs (format: idA, idB)", type=['csv'])
        elif splitted_pairs == ALREADY_SPLIT_OPTION:
            train_pairs = st.file_uploader("Upload the train pairs (format: idA, idB)", type=['csv'])
            val_pairs = st.file_uploader("Upload the validation pairs (format: idA, idB)", type=['csv'])
            test_pairs = st.file_uploader("Upload the test pairs (format: idA, idB)", type=['csv'])
    else:
        splitted_pairs = st.selectbox(
            "Select the format of the your true matches",
            [TESTING_PAIR_OPTION, NO_TRUE_MATCHES_OPTION]
        )

        if splitted_pairs == TESTING_PAIR_OPTION:
            test_pairs = st.file_uploader("Upload the test pairs (format: idA, idB)", type=['csv'])

    st.session_state.dataset_name = CUSTOM_DATASET_NAME
    st.session_state.known_pairs = any(
        file.startswith("gs_") for file in os.listdir(os.path.join(DATA_FOLDER, st.session_state.dataset_name)))

    if st.button("Upload dataset"):
        if (not train_pairs or not val_pairs or not test_pairs) and splitted_pairs == ALREADY_SPLIT_OPTION:
            st.warning("Please upload all pairs")
        elif not train_pairs and splitted_pairs == NOT_SPLIT_OPTION:
            st.warning("Please upload the train pairs")
        elif not test_pairs and splitted_pairs == TESTING_PAIR_OPTION:
            st.warning("Please upload the test pairs")
        elif tableA is not None and tableB is not None:
            save_tables(tableA, tableB, DATA_FOLDER, train_pairs=train_pairs, val_pairs=val_pairs,
                        test_pairs=test_pairs)
            st.write("Dataset uploaded successfully")
        else:
            st.warning("Please upload both tables")

######## Blocking ########

st.header("Model")
st.subheader("Blocking")

pair_used = st.radio(
    "Select the pairs to use",
    ["Already included pairs", BLOCKED_PAIRS_OPTION] if st.session_state.known_pairs else [BLOCKED_PAIRS_OPTION]
)

if pair_used == BLOCKED_PAIRS_OPTION:
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

            table_a_serialized, table_b_serialized, X_train_ids, y_train, X_valid_ids, y_valid, X_test_ids, y_test = load_data(
                os.path.join(DATA_FOLDER, st.session_state.dataset_name), remove_col_names=remove_col_name,
                order_cols=order_cols)

            all_true_matches = merge_true_matches(X_train_ids, y_train, X_valid_ids, y_valid, X_test_ids, y_test)

            st.write("True matches merged")
            st.write("Starting blocking")

            blocked_pairs = perform_blocking_sbert(blocking, table_a_serialized, table_b_serialized,
                                                   n_neighbors=n_neighbors, metric='cosine', device=device)
            if merge_with_tfidf:
                blocked_pairs = merge_indices(blocked_pairs,
                                              perform_blocking_tfidf(table_a_serialized, table_b_serialized,
                                                                     n_neighbors=n_neighbors, metric='cosine'))
                st.write("Merged with TF-IDF")

            st.write("Blocking done")

            st.session_state.blocked_pairs = blocked_pairs

            reduction_ratio, recall, f1 = get_blocking_metrics(blocked_pairs, all_true_matches, len(table_a_serialized),
                                                               len(table_b_serialized)) if len(
                all_true_matches) > 0 else (None, None, None)

            st.session_state.blocking_metrics["reduction_ratio"] = reduction_ratio
            st.session_state.blocking_metrics["recall"] = recall
            st.session_state.blocking_metrics["f1"] = f1
else:
    st.session_state.blocked_pairs = None
    st.session_state.blocking_metrics = {
        "reduction_ratio": None,
        "recall": None,
        "f1": None
    }

# Display metrics from session state
if st.session_state.blocking_metrics["reduction_ratio"] is not None:
    st.write(f"Reduction ratio: {st.session_state.blocking_metrics['reduction_ratio']}")
    st.write(f"Recall: {st.session_state.blocking_metrics['recall']}")
    st.write(f"F1 score: {st.session_state.blocking_metrics['f1']}")

######## Matching ########

st.subheader("Matching")

columns = st.columns(2)
with columns[0]:
    remove_col_name = st.checkbox("Remove column name (recommended)", key="remove_col_name_matching", value=True)
with columns[1]:
    order_cols = st.checkbox("Order columns (recommended)", key="order_cols_matching", value=True)

if st.session_state.supervised:
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

            train_loader, valid_set, y_valid, test_set, y_test = prepare_data_cross_encoder(
                os.path.join(DATA_FOLDER, st.session_state.dataset_name), remove_col_names=remove_col_name,
                order_cols=order_cols, blocked_pairs=st.session_state.blocked_pairs)

            st.write("Data prepared")

            logits, train_time = fit_cross_encoder(model_name, train_loader, valid_set, y_valid, test_set,
                                                   epochs=EPOCHS, learning_rate=LEARNING_RATE,
                                                   weight_decay=WEIGHT_DECAY, device=device)

            st.write("Model fitted")

            accuracy, precision, recall, f1, roc_auc = evaluate_cross_encoder(logits, y_test, threshold=THRESHOLD)

            st.write("Matching done")

            st.session_state.cross_encoder_metrics["accuracy"] = accuracy
            st.session_state.cross_encoder_metrics["precision"] = precision
            st.session_state.cross_encoder_metrics["recall"] = recall
            st.session_state.cross_encoder_metrics["f1"] = f1
            st.session_state.cross_encoder_metrics["roc_auc"] = roc_auc

        if st.session_state.cross_encoder_metrics["accuracy"] is not None:
            # Print an example of the predictions
            # Check the first 1 prediction and first 0 prediction
            st.write("Example of predictions")
            count_1, count_0 = 0, 0
            for i in range(len(test_set)):
                if y_test[i] == 1 and count_1 < 1:
                    st.write(f"Prediction: {logits[i]} - True value: {y_test[i]}")
                    st.write("Example of prediction 1:")
                    st.write(f"Entity A: {test_set[i][0]}")
                    st.write(f"Entity B: {test_set[i][1]}")
                    count_1 += 1
                elif y_test[i] == 0 and count_0 < 1:
                    st.write(f"Prediction: {logits[i]} - True value: {y_test[i]}")
                    st.write("Example of prediction 0:")
                    st.write(f"Entity A: {test_set[i][0]}")
                    st.write(f"Entity B: {test_set[i][1]}")
                    count_0 += 1
                if count_1 == 1 and count_0 == 1:
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

    model_name_zero_shot = st.selectbox(
        "Select the model to use",
        [
            'all-mpnet-base-v2',
            'sentence-transformers/multi-qa-mpnet-base-dot-v1',
            'facebook/bart-large-mnli',
        ],
        index=1
    )

    if st.button("Run matching"):
        if unsupervised_method == 'ZeroShot Embedding':

            with st.status("Matching in progress..."):
                table_a_serialized, table_b_serialized, X_train_ids, y_train, X_valid_ids, y_valid, X_test_ids, y_test = load_data(
                    os.path.join(DATA_FOLDER, st.session_state.dataset_name), remove_col_names=remove_col_name,
                    order_cols=order_cols)
                similarity_matrix_test = None
                X1_test, X2_test = None, None

                if st.session_state.known_pairs:
                    X1_test, X2_test = [table_a_serialized[i[0]] for i in X_test_ids], [table_b_serialized[i[1]] for i
                                                                                        in X_test_ids]

                elif st.session_state.blocked_pairs is not None:
                    all_matches = []
                    for i in range(len(table_a_serialized)):
                        for j in blocked_pairs[i]:
                            all_matches.append((i, j))

                    X1_test, X2_test = [table_a_serialized[i[0]] for i in all_matches], [table_b_serialized[i[1]] for i
                                                                                         in all_matches]

                else:
                    st.write("Please run the blocking first")
                    st.stop()

                model = SentenceTransformer(model_name_zero_shot, device=device)

                embeddings1_test = model.encode(X1_test)
                embeddings2_test = model.encode(X2_test)

                st.write("Embeddings done")

                similarity_matrix_test = cosine_similarity(embeddings1_test, embeddings2_test)

                st.write("Similarity matrix done")

                if similarity_matrix_test is not None:
                    accuracy, precision, recall, f1, roc_auc = evaluate_zero_shot(similarity_matrix_test, y_test,
                                                                                  threshold=THRESHOLD)

                    st.session_state.unsupervised_metrics["accuracy"] = accuracy
                    st.session_state.unsupervised_metrics["precision"] = precision
                    st.session_state.unsupervised_metrics["recall"] = recall
                    st.session_state.unsupervised_metrics["f1"] = f1
                    st.session_state.unsupervised_metrics["roc_auc"] = roc_auc

                st.write("Matching done")

        else:
            ### Few shots
            pass

    # Display metrics from session state
    if st.session_state.unsupervised_metrics["accuracy"] is not None:
        st.write(f"Accuracy: {st.session_state.unsupervised_metrics['accuracy']}")
        st.write(f"Precision: {st.session_state.unsupervised_metrics['precision']}")
        st.write(f"Recall: {st.session_state.unsupervised_metrics['recall']}")
        st.write(f"F1 score: {st.session_state.unsupervised_metrics['f1']}")
        st.write(f"ROC AUC: {st.session_state.unsupervised_metrics['roc_auc']}")

######## Evaluation ########

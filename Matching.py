import os

import pandas as pd
import streamlit as st
import torch
from sentence_transformers import (InputExample,
                                   SentenceTransformer,
                                   losses)
from sentence_transformers.util import pairwise_cos_sim
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

import src.import_data as import_data
import src.session_state as session_state
from model.Blocking import (get_blocking_metrics, merge_indices,
                            merge_true_matches, perform_blocking_sbert,
                            perform_blocking_tfidf)
from model.CrossEncoderModel import (fit_cross_encoder,
                                     prepare_data_cross_encoder)
from model.ZeroShotModel import evaluate_zero_shot
from model.utils import load_data, save_tables

######## Constants ########

DATA_FOLDER = import_data.DATA_FOLDER
RESULTS_FOLDER = "results"
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

def reset_training_pairs():
    st.session_state.training_pairs_few_shot = None

session_state.initialize_session_state()

if 'supervised_model' not in st.session_state:
    st.session_state.supervised_model = True

if 'testing_mode' not in st.session_state:
    st.session_state.testing_mode = False

if 'known_pairs' not in st.session_state:
    st.session_state.known_pairs = False

if 'training_pairs_few_shot' not in st.session_state:
    st.session_state.training_pairs_few_shot = None

if 'blocked_pairs' not in st.session_state:
    st.session_state.blocked_pairs = None

if 'matching_pairs' not in st.session_state:
    st.session_state.matching_pairs = []

######## Streamlit app ########

st.set_page_config(
    page_title="WavER Matching",
    page_icon="🔀",
    layout="wide",
)

st.sidebar.title("Presentation")
st.sidebar.markdown(
    """
    ## About this App

    This app is a demonstration of the entity resolution project developed by **Tristan PERROT** for his degree project at KTH in collaboration with Wavestone.

    The goal of this project is to provide a tool to perform entity resolution using different methods such as supervised learning, unsupervised learning, and few-shot learning.

    ### Sections:
    1. **Dataset**: Upload a dataset to perform entity resolution.
    2. **Model**: Select the method to use for entity resolution.
    3. **Blocking**: Perform blocking to reduce the number of pairs to compare.
    4. **Matching**: Compare the pairs and evaluate the results.
    5. **Results**: Download the results of the entity resolution.

    The app is designed to be user-friendly and easy to use. If you have any questions or feedback, please feel free to contact me at [tristan.perrot@wavestone.com](mailto:tristan.perrot@wavestone.com).
    """
)

st.title("WavER - Entity Resolution Project")
st.write("Tristan PERROT degree project for KTH @ Wavestone")

st.session_state.supervised_model = st.selectbox(
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
        import_data.DATASET_SUFFIX.keys(), index=2, on_change=reset_training_pairs
    )

    st.session_state.dataset_name = dataset_name

    if st.button("Download dataset"):
        downloaded = import_data.download_dataset(dataset_name)
        if downloaded:
            st.success("Dataset downloaded successfully")
        else:
            st.error(
                "Dataset already exists, skipping download. If you want to re-download, please delete the folder first.")

    if not (os.path.exists(os.path.join(DATA_FOLDER, dataset_name))):
        st.warning("Dataset not found, please download it first")
        st.stop()

else:
    tableA = st.file_uploader("Upload the table A (format: subject_id, col1, col2,..)", type=['csv'])
    tableB = st.file_uploader("Upload the table B (format: subject_id, col1, col2,..)", type=['csv'])

    train_pairs, val_pairs, test_pairs = None, None, None

    if st.session_state.supervised_model:
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
            st.success("Dataset uploaded successfully")
        else:
            st.warning("Please upload both tables")

######## Blocking ########

st.header("Model")
st.subheader("Blocking")

if st.session_state.dataset_name:
    
    try:
        st.session_state.known_pairs = any(
            file.startswith("gs_") for file in os.listdir(os.path.join(DATA_FOLDER, st.session_state.dataset_name)))

        st.session_state.cols_a, st.session_state.cols_b = load_data(
            os.path.join(DATA_FOLDER, st.session_state.dataset_name), return_only_col_names=True)

        # Remove id column
        st.session_state.cols_a = st.session_state.cols_a[1:]
        st.session_state.cols_b = st.session_state.cols_b[1:]
    except FileNotFoundError:
        st.warning("Please upload a dataset first")
        st.stop()

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
        remove_col_name = st.checkbox("Remove column name in serialization (recommended)",
                                      key="remove_col_name_blocking", value=True)
    with columns[2]:
        order_cols = st.checkbox("Order columns (not recommended)", key="order_cols_blocking")

    columns = st.columns(2)
    with columns[0]:
        cols_a_to_rm = st.multiselect("Select the columns to remove in table A", st.session_state.cols_a,
                                      key="cols_a_to_rm_blocking")
    with columns[1]:
        cols_b_to_rm = st.multiselect("Select the columns to remove in table B", st.session_state.cols_b,
                                      key="cols_b_to_rm_blocking")

    n_neighbors = st.number_input("Number of neighbors to retrieve", value=15, min_value=1, max_value=40)

    if st.button("Run blocking"):
        with st.status("Blocking...", expanded=True):
            st.write("Blocking started")

            table_a_serialized, table_b_serialized, X_train_ids, y_train, X_valid_ids, y_valid, X_test_ids, y_test = load_data(
                os.path.join(DATA_FOLDER, st.session_state.dataset_name), remove_col_names=remove_col_name,
                order_cols=order_cols, cols_a_to_rm=cols_a_to_rm, cols_b_to_rm=cols_b_to_rm)

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
    remove_col_name = st.checkbox("Remove column name in serialization  (recommended)", key="remove_col_name_matching",
                                  value=True)
    cols_a_to_rm = st.multiselect("Select the columns to remove in table A", st.session_state.cols_a,
                                  key="cols_a_to_rm_matching")
with columns[1]:
    order_cols = st.checkbox("Order columns (recommended)", key="order_cols_matching", value=True)
    cols_b_to_rm = st.multiselect("Select the columns to remove in table B", st.session_state.cols_b,
                                  key="cols_b_to_rm_matching")

if st.session_state.supervised_model:
    model_name = st.selectbox(
        "Select the model to use",
        [
            'cross-encoder/stsb-roberta-base',
            'cross-encoder/stsb-distilroberta-base',
            'cross-encoder/ms-marco-MiniLM-L-12-v2',
        ],
        index=1
    )
    
    columns = st.columns(3)
    with columns[0]:
        EPOCHS = st.number_input("Number of epochs", value=1, min_value=1, max_value=10)
    with columns[1]:
        LEARNING_RATE = st.number_input("Learning rate", value=2e-5, min_value=1e-6, max_value=1e-2, format="%.6f")
    with columns[2]:
        WEIGHT_DECAY = st.number_input("Weight decay", value=0.01, min_value=0.0, max_value=1.0)
    
    if st.button("Run matching"):
        with st.status("Matching in progress...", expanded=True):
            st.write("Matching started")

            train_loader, valid_set, y_valid, test_set, X_test_ids, y_test = prepare_data_cross_encoder(
                os.path.join(DATA_FOLDER, st.session_state.dataset_name), remove_col_names=remove_col_name,
                order_cols=order_cols, blocked_pairs=st.session_state.blocked_pairs, cols_b_to_rm=cols_b_to_rm,
                cols_a_to_rm=cols_a_to_rm)

            st.write("Data prepared")

            logits, train_time = fit_cross_encoder(model_name, train_loader, valid_set, y_valid, test_set,
                                                   epochs=EPOCHS, learning_rate=LEARNING_RATE,
                                                   weight_decay=WEIGHT_DECAY, device=device)

            output = [(X_test_ids[i][0], X_test_ids[i][1], logits[i]) for i in range(len(X_test_ids))]
            st.session_state.output = output

            st.write("Model fitted")

        st.success("Matching done")

else:

    few_shot_method = st.selectbox(
        "Select the method to use",
        ["Zero-shot", "Few-shot"],
        on_change=reset_training_pairs,
        index=0
    ) == "Few-shot"

    model_name_zero_shot = st.selectbox(
        "Select the model to use",
        [
            'all-mpnet-base-v2',
            'sentence-transformers/multi-qa-mpnet-base-dot-v1',
            'facebook/bart-large-mnli',
        ],
        index=1
    )

    columns = st.columns(2)
    with columns[0]:
        THRESHOLD = st.number_input("Threshold for matching", value=0.7, min_value=0.0, max_value=1.0, format="%.2f")

    if few_shot_method:

        with columns[1]:
            num_epochs_few_shot = st.number_input("Number of epochs for few-shot learning", value=5, min_value=1, max_value=10)

        upload_training_pairs = st.selectbox(
            "Select the training pairs",
            ["Upload training pairs", "Create here"],
            on_change=reset_training_pairs
        ) == "Upload training pairs"

        if upload_training_pairs:
            # Either upload training pairs or select from the dataset by searching in it
            training_pairs = st.file_uploader("Upload the training pairs (format: idA, idB)", type=['csv'])

            if training_pairs is not None:
                training_pairs = pd.read_csv(training_pairs)
                # Save the training pairs in the session state
                st.session_state.training_pairs_few_shot = training_pairs
                print(training_pairs)

        else:
            st.write("Select the training pairs from the tables")

            table_a_serialized, table_b_serialized, X_train_ids, y_train, X_valid_ids, y_valid, X_test_ids, y_test = load_data(
                os.path.join(DATA_FOLDER, st.session_state.dataset_name), remove_col_names=remove_col_name,
                order_cols=order_cols)

            col1, col2 = st.columns(2)
            with col1:
                search_term_a = st.text_input("Enter search term to find identical pairs in table A")
            with col2:
                search_term_b = st.text_input("Enter search term to find identical pairs in table B")

            if search_term_a and search_term_b:
                new_matching_row_a = [i for i, row in enumerate(table_a_serialized) if search_term_a in row]
                new_matching_row_b = [i for i, row in enumerate(table_b_serialized) if search_term_b in row]

                if new_matching_row_a and new_matching_row_b:

                    with col1:
                        st.write("Table A")
                        for i, index_a in enumerate(new_matching_row_a[:10]):
                            st.write(f"{i}: {table_a_serialized[index_a]}")

                    with col2:
                        st.write("Table B")
                        for i, index_b in enumerate(new_matching_row_b[:10]):
                            st.write(f"{i}: {table_b_serialized[index_b]}")

                    st.write(
                        f"Found {len(new_matching_row_a)} matching rows in table A and {len(new_matching_row_b)} in table B")
                    col1, col2 = st.columns(2)

                    with col1:
                        selected_indices_a = st.selectbox("Select the indice in table A",
                                                          range(len(new_matching_row_a)), index=0)
                    with col2:
                        selected_indices_b = st.selectbox("Select the indice in table B",
                                                          range(len(new_matching_row_b)), index=0)
                    label = st.selectbox("Select the label", [0, 1], index=0)

                    if st.button("Add pair"):
                        if st.session_state.training_pairs_few_shot is None:
                            st.session_state.training_pairs_few_shot = pd.DataFrame(columns=["idA", "idB", "label"])
                        st.session_state.training_pairs_few_shot = pd.concat([st.session_state.training_pairs_few_shot,
                                                                              pd.DataFrame([[new_matching_row_a[
                                                                                                 selected_indices_a],
                                                                                             new_matching_row_b[
                                                                                                 selected_indices_b],
                                                                                             label]],
                                                                                           columns=["idA", "idB",
                                                                                                    "label"])])
                        print(st.session_state.training_pairs_few_shot)
                    st.write("Actual training pairs:")
                    # Print but write the actual training pairs and not the indices in a Dataframe format
                    df_training_pairs = pd.DataFrame(columns=["Entity A", "Entity B", "Label"])
                    if st.session_state.training_pairs_few_shot is not None:
                        for i, row in st.session_state.training_pairs_few_shot.iterrows():
                            df_training_pairs = pd.concat([df_training_pairs, pd.DataFrame([[table_a_serialized[
                                                                                                 int(row["idA"])],
                                                                                             table_b_serialized[
                                                                                                 int(row["idB"])],
                                                                                             row["label"]]],
                                                                                           columns=["Entity A",
                                                                                                    "Entity B",
                                                                                                    "Label"])])
                    if st.button("Reset training pairs"):
                        st.session_state.training_pairs_few_shot = None
                        st.write("Training pairs reset")
                    else:
                        st.write(df_training_pairs)
                else:
                    st.write("No matching pairs found")

    if st.button("Run matching"):
        with st.status("Matching in progress...", expanded=True):
            similarity_matrix_test = None
            X1_test, X2_test = None, None

            table_a_serialized, table_b_serialized, X_train_ids, y_train, X_valid_ids, y_valid, X_test_ids, y_test = load_data(
                os.path.join(DATA_FOLDER, st.session_state.dataset_name), remove_col_names=remove_col_name,
                order_cols=order_cols)

            if st.session_state.known_pairs:
                X1_test, X2_test = [table_a_serialized[i[0]] for i in X_test_ids], [table_b_serialized[i[1]] for i
                                                                                    in X_test_ids]

            elif st.session_state.blocked_pairs is not None:
                X_test_ids = []
                for i in range(len(table_a_serialized)):
                    for j in blocked_pairs[i]:
                        X_test_ids.append((i, j))

                X1_test, X2_test = [table_a_serialized[i[0]] for i in X_test_ids], [table_b_serialized[i[1]] for i
                                                                                    in X_test_ids]

            else:
                st.write("Please run the blocking first")
                st.stop()

            model = SentenceTransformer(model_name_zero_shot, device=device)

            if few_shot_method and st.session_state.training_pairs_few_shot is None:
                st.write("Please upload the training pairs")
                st.stop()
            elif few_shot_method:
                cols = st.session_state.training_pairs_few_shot.columns
                X1_train_ids, X2_train_ids = st.session_state.training_pairs_few_shot[cols[0]].values, \
                st.session_state.training_pairs_few_shot[cols[1]].values
                print(X1_train_ids, X2_train_ids)
                print(len(table_a_serialized), len(table_b_serialized))
                X1_train, X2_train = [table_a_serialized[i - len(table_b_serialized)] for i in X1_train_ids], [
                    table_b_serialized[i] for i in X2_train_ids]
                y_train = st.session_state.training_pairs_few_shot[cols[2]].values

                embeddings1_train = model.encode(X1_train)
                embeddings1_train = model.encode(X2_train)

                st.write("Embeddings done")

                # Prepare the dataset for training
                train_examples = [InputExample(texts=[X1_train[i], X2_train[i]], label=y_train[i]) for i in
                                  range(len(X1_train))]

                # Convert to a dataset suitable for Sentence Transformers
                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16, num_workers=0)

                # Define the loss function
                train_loss = losses.CosineSimilarityLoss(model=model)

                # Training parameters
                warmup_steps = int(len(train_dataloader) * num_epochs_few_shot * 0.1)  # 10% of training steps for warm-up

                # Define a directory to save the model
                model_save_path = "output/trained_sentence_transformer_model"

                # Train the model
                model.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    epochs=num_epochs_few_shot,
                    warmup_steps=warmup_steps,
                    output_path=model_save_path
                )

                st.write("Model trained")

            embeddings1_test = model.encode(X1_test)
            embeddings2_test = model.encode(X2_test)

            st.write("Embeddings done")

            similarity_test = pairwise_cos_sim(embeddings1_test, embeddings2_test)

            st.write("Similarity done")

            if similarity_test is not None:
                accuracy, precision, recall, f1, roc_auc = evaluate_zero_shot(similarity_test, y_test,
                                                                              threshold=THRESHOLD)

                output = [(X_test_ids[i][0], X_test_ids[i][1], similarity_test[i]) for i in
                          range(len(X_test_ids))]
                st.session_state.output = output
            st.write("Matching done")

        st.success("Matching done")

######## Evaluation ########

st.header("Result export")
if st.session_state.output:
    output_df = pd.DataFrame(st.session_state.output, columns=['Entity A', 'Entity B', 'Score'])

    st.session_state.output_df = output_df

    # Convert DataFrame to CSV
    csv_data = output_df.to_csv(index=False)

    # Provide download button
    st.download_button(
        label="Download results as CSV",
        data=csv_data,
        file_name="results_" + st.session_state.dataset_name + ".csv",
        mime="text/csv",
    )
else:
    st.warning("No results to download. Please run the matching process first.")

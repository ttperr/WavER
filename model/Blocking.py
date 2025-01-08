import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm.notebook import tqdm, trange

def merge_true_matches(X_train_ids, y_train, X_valid_ids, y_valid, X_test_ids, y_test):
    """
    Merges true matches from training, validation, and test datasets.

    This function iterates through the provided training, validation, and test datasets,
    and collects all pairs of IDs that are marked as true matches (i.e., where the corresponding
    label is 1). The resulting set contains unique pairs of IDs that are true matches across
    all datasets.

    Args:
        X_train_ids (list of tuples): List of ID pairs for the training dataset.
        y_train (list of int): List of labels for the training dataset, where 1 indicates a true match.
        X_valid_ids (list of tuples): List of ID pairs for the validation dataset.
        y_valid (list of int): List of labels for the validation dataset, where 1 indicates a true match.
        X_test_ids (list of tuples): List of ID pairs for the test dataset.
        y_test (list of int): List of labels for the test dataset, where 1 indicates a true match.

    Returns:
        set: A set of unique ID pairs that are true matches across the training, validation, and test datasets.
    """
    all_true_matches = set()
    for i in range(len(X_train_ids)):
        if y_train[i] == 1:
            all_true_matches.add((X_train_ids[i][0], X_train_ids[i][1]))
    for i in range(len(X_valid_ids)):
        if y_valid[i] == 1:
            all_true_matches.add((X_valid_ids[i][0], X_valid_ids[i][1]))
    for i in range(len(X_test_ids)):
        if y_test[i] == 1:
            all_true_matches.add((X_test_ids[i][0], X_test_ids[i][1]))
    return all_true_matches

def perform_blocking_sbert(model_name, table_a_serialized, table_b_serialized, n_neighbors=20, metric='cosine', device='cpu'):
    """
    Perform blocking using Sentence-BERT embeddings and k-nearest neighbors.

    Args:
        model_name (str): The name of the pre-trained Sentence-BERT model to use.
        table_a_serialized (list of str): The serialized data from table A to be encoded.
        table_b_serialized (list of str): The serialized data from table B to be encoded.
        n_neighbors (int, optional): The number of nearest neighbors to find. Defaults to 20.
        metric (str, optional): The distance metric to use for the k-nearest neighbors algorithm. Defaults to 'cosine'.
        device (str, optional): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        numpy.ndarray: Indices of the nearest neighbors in table B for each entry in table A.
    """
    model = SentenceTransformer(model_name, device=device)
    print("Model loaded")
    table_a_embeddings = model.encode(table_a_serialized)
    print("Table A encoded")
    table_b_embeddings = model.encode(table_b_serialized)
    print("Table B encoded")
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn.fit(table_b_embeddings)
    print("k-NN model fitted")

    return knn.kneighbors(table_a_embeddings, return_distance=False)

def get_blocking_metrics(indices, all_true_matches, len_table_a, len_table_b):
    """
    Calculate blocking metrics including reduction ratio, recall, and F1 score.
    Parameters:
    indices (list of lists): A list where each element is a list of indices representing blocked pairs.
    all_true_matches (set of tuples): A set of tuples where each tuple represents a true match (i, j).
    len_table_a (int): The number of records in table A.
    len_table_b (int): The number of records in table B.
    Returns:
    tuple: A tuple containing the reduction ratio, recall, and F1 score.
    """
    n_m = len(all_true_matches)
    n_n = len_table_a * len_table_b - n_m
    s_m = 0
    s_n = 0

    for i in range(len(indices)):
        for j in indices[i]:
            if (i, j) in all_true_matches:
                s_m += 1
            else:
                s_n += 1
    
    reduction_ratio = 1 - (s_m + s_n) / (n_m + n_n)
    recall = s_m / n_m
    f1 = 2 * (reduction_ratio * recall) / (reduction_ratio + recall)

    return reduction_ratio, recall, f1

def perform_blocking_tfidf(table_a_serialized, table_b_serialized, n_neighbors=20, metric='cosine'):
    """
    Perform blocking using TF-IDF vectorization and k-nearest neighbors.

    This function takes two serialized tables, vectorizes them using TF-IDF, and then
    finds the k-nearest neighbors for each entry in the first table from the second table.

    Args:
        table_a_serialized (list of str): The first table serialized as a list of strings.
        table_b_serialized (list of str): The second table serialized as a list of strings.
        n_neighbors (int, optional): The number of neighbors to use for k-nearest neighbors. Default is 20.
        metric (str, optional): The distance metric to use for k-nearest neighbors. Default is 'cosine'.

    Returns:
        numpy.ndarray: An array of indices of the nearest neighbors in table_b for each entry in table_a.
    """
    vectorizer = TfidfVectorizer()
    table_a_tfidf = vectorizer.fit_transform(table_a_serialized)
    print("Table A vectorized")
    table_b_tfidf = vectorizer.transform(table_b_serialized)
    print("Table B vectorized")
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn.fit(table_b_tfidf)
    print("k-NN model fitted")
    return knn.kneighbors(table_a_tfidf, return_distance=False)

def merge_indices(indices1, indices2):
    """
    Merges two lists of index lists by performing a union of corresponding index lists.

    Args:
        indices1 (list of list of int): The first list of index lists.
        indices2 (list of list of int): The second list of index lists.

    Returns:
        list of list of int: A new list where each element is the union of the corresponding
                             index lists from indices1 and indices2.

    Raises:
        AssertionError: If the lengths of indices1 and indices2 are not equal.
    """
    merged_indices = []
    assert len(indices1) == len(indices2)
    for i in range(len(indices1)):
        merged_indices.append(list(set(indices1[i]) | set(indices2[i])))
    return merged_indices
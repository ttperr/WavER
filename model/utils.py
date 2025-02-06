#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File name: utils.py
Author: Tristan PERROT
Created: 10/10/2024
Version: 1.0
Description: Utils functions for Entity Resolution project
"""

import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def get_entity(id, table):
    """
    Get an entity from a pandas DataFrame based on its ID.

    :param id: The ID of the entity to retrieve. The ID is expected to be in the format '[ab]X' where X is the index of the entity in the DataFrame.
    :type id: str
    :param table: The pandas DataFrame containing the entities.
    :type table: pd.DataFrame

    :return: The entity corresponding to the given ID.
    :rtype: pd.Series
    """

    real_id = int(id[1:])
    return table.iloc[real_id]


def save_tables(table_a, table_b, data_dir, train_pairs=None, val_pairs=None, test_pairs=None):
    """
    Save tables and optionally split and save train, validation, and test pairs.
    
    Parameters
    ----------
    table_a : file-like
        The first table to save.
    table_b : file-like
        The second table to save.
    data_dir : str
        The directory where the tables and pairs will be saved.
    train_pairs : file-like, optional
        The training pairs.
    val_pairs : file-like, optional
        The validation pairs.
    test_pairs : file-like, optional
        The test pairs.

    Returns
    -------

    The function saves `table_a` and `table_b` as "1_custom.csv" and "2_custom.csv" respectively
    in a subdirectory named "custom_dataset" within `data_dir`. If `train_pairs` is provided and
    `val_pairs` and `test_pairs` are not provided, it splits the training pairs into training,
    validation, and test sets and saves them as "gs_train.csv", "gs_val.csv", and "gs_test.csv".
    If only `test_pairs` is provided, it saves it as "gs_test.csv". If all three pairs are provided,
    it saves them directly as "gs_train.csv", "gs_val.csv", and "gs_test.csv".
    """
    if not os.path.exists(os.path.join(data_dir, "custom_dataset")):
        os.makedirs(os.path.join(data_dir, "custom_dataset"))
    with open(os.path.join(data_dir, "custom_dataset", "1_custom.csv"), "wb") as f:
        f.write(table_a.read())
    with open(os.path.join(data_dir, "custom_dataset", "2_custom.csv"), "wb") as f:
        f.write(table_b.read())

    if train_pairs is not None and val_pairs is None and test_pairs is None:
        pairs_df = pd.read_csv(train_pairs)
        pairs_train, pairs_val_test = train_test_split(pairs_df, test_size=0.3, random_state=42)
        pairs_val, pairs_test = train_test_split(pairs_val_test, test_size=0.5, random_state=42)
        pairs_train.to_csv(os.path.join(data_dir, "custom_dataset", "gs_train.csv"), index=False)
        pairs_val.to_csv(os.path.join(data_dir, "custom_dataset", "gs_val.csv"), index=False)
        pairs_test.to_csv(os.path.join(data_dir, "custom_dataset", "gs_test.csv"), index=False)

    elif train_pairs is None and val_pairs is None and test_pairs is not None:
        pars_df = pd.read_csv(test_pairs)
        pars_df.to_csv(os.path.join(data_dir, "custom_dataset", "gs_test.csv"), index=False)

    elif train_pairs is not None and val_pairs is not None and test_pairs is not None:
        pairs_train = pd.read_csv(train_pairs)
        pairs_val = pd.read_csv(val_pairs)
        pairs_test = pd.read_csv(test_pairs)
        pairs_train.to_csv(os.path.join(data_dir, "custom_dataset", "gs_train.csv"), index=False)
        pairs_val.to_csv(os.path.join(data_dir, "custom_dataset", "gs_val.csv"), index=False)
        pairs_test.to_csv(os.path.join(data_dir, "custom_dataset", "gs_test.csv"), index=False)

    print('Tables saved successfully')


def serialize_entities(entity1, entity2=None, remove_col_names=False):
    """
    Serializes one or two entities into a single string representation.

    This function takes one or two pandas Series objects representing entities and converts them into a
    string format. The resulting string includes column names and values, separated by special tokens.
    If two entities are provided, they are separated by a [SEP] token.

    Parameters:
    -----------
    entity1 : pd.Series
        The first entity to serialize, represented as a pandas Series.
    entity2 : pd.Series, optional
        The second entity to serialize, represented as a pandas Series. If not provided,
        only the first entity will be serialized. Default is None.
    remove_col_names : bool, optional
        If True, column names will be omitted from the serialized string, leaving only the values.
        If False, both column names and values will be included. Default is False.

    Returns: -------- str A serialized string representation of the entity or entities. The format is: '[COL] column1
    [VAL] value1 [COL] column2 [VAL] value2 ... [SEP] [COL] column1 [VAL] value1 [COL] column2 [VAL] value2 ...' If
    remove_col_names is True, the format will be: 'value1 value2 ... [SEP] value1 value2 ...'

    Notes:
    ------
    - NaN values in the input Series are replaced with empty strings.
    - If only one entity is provided, the [SEP] token and the second part of the string will be omitted.
    """
    entity1_nan = entity1.fillna('')
    entity2_nan = entity2.fillna('') if entity2 is not None else pd.Series(dtype='str')
    if remove_col_names:
        e1_string = " ".join([f'{val}' for val in entity1_nan])
        e2_string = " ".join([f'{val}' for val in entity2_nan]) if entity2 is not None else ""
    else:
        e1_string = " ".join([f'[COL] {col} [VAL] {val}' for col, val in zip(entity1_nan.index, entity1_nan)])
        e2_string = " ".join([f'[COL] {col} [VAL] {val}' for col, val in
                              zip(entity2_nan.index, entity2_nan)]) if entity2 is not None else ""
    return e1_string + ' [SEP] ' + e2_string if entity2 is not None else e1_string


def deserialize_entities(input_string):
    """
    Deserialize a string representation of two entities into two pandas DataFrames.

    The input string is expected to have the following format: '[COL] column1 [VAL] value1 [COL] column2 [VAL] value2
    ... [SEP] [COL] column1 [VAL] value1 [COL] column2 [VAL] value2 ...'

    :param input_string: The string representation of the entities.
    :type input_string: str

    :returns: A tuple containing two pandas DataFrames (each representing an entity).
    :rtype: tuple
    """
    # Split the input string into entities based on [CLS]
    entities = input_string.split(' [SEP] ')

    # Split the entities into columns and values
    e1 = entities[0].split('[COL] ')[1:]
    e2 = entities[1].split('[COL] ')[1:] if len(entities) > 1 else []

    e1 = [col.split('[VAL] ') for col in e1]
    e2 = [col.split('[VAL] ') for col in e2] if len(e2) > 0 else []

    e1_df = pd.DataFrame(e1, columns=['column', 'value'])
    e2_df = pd.DataFrame(e2, columns=['column', 'value']) if len(e2) > 0 else pd.DataFrame()

    # Transpose the DataFrames
    e1_df = e1_df.T
    e2_df = e2_df.T if len(e2_df) > 0 else pd.DataFrame()

    e1_df.columns = e1_df.iloc[0]
    e1_df = e1_df.drop(e1_df.index[0])
    e1_df.reset_index(drop=True, inplace=True)

    if len(e2_df) > 0:
        e2_df.columns = e2_df.iloc[0]
        e2_df = e2_df.drop(e2_df.index[0])
        e2_df.reset_index(drop=True, inplace=True)

    return e1_df, e2_df


def load_data(data_dir, cols_a_to_rm=None, cols_b_to_rm=None, order_cols=False, remove_col_names=False, return_tables=False, return_only_col_names=False, verbose=False):
    """
    Load and preprocess data from the specified directory.
    
    Parameters:
    ----------
    data_dir (str): Directory containing the data files.
    cols_a_to_rm (list, optional): List of columns to remove from table A. Defaults to None.
    cols_b_to_rm (list, optional): List of columns to remove from table B. Defaults to None.
    order_cols (bool, optional): Whether to reorder columns based on the length of their values. Defaults to False.
    remove_col_names (bool, optional): Whether to remove column names during serialization. Defaults to False.
    custom_dataset (bool, optional): Whether to use custom dataset files. Defaults to False.
    
    Returns:
    --------
    tuple: A tuple containing:
        - table_a_serialized (list): Serialized entities from table A.
        - table_b_serialized (list): Serialized entities from table B.
        - X_train (list): Training data pairs.
        - y_train (list): Training data labels.
        - X_valid (list): Validation data pairs.
        - y_valid (list): Validation data labels.
        - X_test (list): Test data pairs.
        - y_test (list): Test data labels.
    """

    pairs_train, pairs_val, pairs_test = None, None, None

    if 'gs_train.csv' in os.listdir(data_dir):
        pairs_train = pd.read_csv(os.path.join(data_dir, 'gs_train.csv'))
    if 'gs_val.csv' in os.listdir(data_dir):
        pairs_val = pd.read_csv(os.path.join(data_dir, 'gs_val.csv'))
    if 'gs_test.csv' in os.listdir(data_dir):
        pairs_test = pd.read_csv(os.path.join(data_dir, 'gs_test.csv'))

    custom_dataset = data_dir.endswith('custom_dataset')

    if not custom_dataset:
        if len([file for file in os.listdir(data_dir) if file.endswith('.csv')]) == 4:
            # Then set 1 is the only file that do not start with gs
            set_1 = [file for file in os.listdir(data_dir) if not file.startswith('gs')][0].split('.')[0]
            set_2 = set_1
            try:
                table_a = pd.read_csv(os.path.join(data_dir, f'{set_1}.csv'), encoding='utf-8')
                table_b = pd.read_csv(os.path.join(data_dir, f'{set_2}.csv'), encoding='utf-8')
            except UnicodeDecodeError:
                table_a = pd.read_csv(os.path.join(data_dir, f'{set_1}.csv'), encoding='latin1')
                table_b = pd.read_csv(os.path.join(data_dir, f'{set_2}.csv'), encoding='latin1')
        else:
            set_1 = data_dir.split(os.sep)[-1].split('-')[0]
            set_2 = data_dir.split(os.sep)[-1].split('-')[1]
            try:
                table_a = pd.read_csv(os.path.join(data_dir, f'1_{set_1}.csv'), encoding='utf-8')
                table_b = pd.read_csv(os.path.join(data_dir, f'2_{set_2}.csv'), encoding='utf-8')
            except UnicodeDecodeError:
                table_a = pd.read_csv(os.path.join(data_dir, f'1_{set_1}.csv'), encoding='latin1')
                table_b = pd.read_csv(os.path.join(data_dir, f'2_{set_2}.csv'), encoding='latin1')
    else:
        table_a = pd.read_csv(os.path.join(data_dir, "1_custom.csv"))
        table_b = pd.read_csv(os.path.join(data_dir, "2_custom.csv"))

    if return_only_col_names:
        return table_a.columns.values, table_b.columns.values

    table_a.rename(columns={'subject_id': 'source_id'}, inplace=True)
    table_b.rename(columns={'subject_id': 'target_id'}, inplace=True)

    # Create dictionaries to map idA and idB to their respective indices
    idA_to_index = dict(zip(table_a['source_id'], table_a.index))
    idB_to_index = dict(zip(table_b['target_id'], table_b.index))

    if pairs_train is not None:
        pairs_train['source_id'] = pairs_train['source_id'].map(idA_to_index)
        pairs_train['target_id'] = pairs_train['target_id'].map(idB_to_index)
    if pairs_val is not None:
        pairs_val['source_id'] = pairs_val['source_id'].map(idA_to_index)
        pairs_val['target_id'] = pairs_val['target_id'].map(idB_to_index)
    if pairs_test is not None:
        pairs_test['source_id'] = pairs_test['source_id'].map(idA_to_index)
        pairs_test['target_id'] = pairs_test['target_id'].map(idB_to_index)

    table_a = table_a.drop(columns=['source_id'])
    table_b = table_b.drop(columns=['target_id'])

    columns_a = pd.DataFrame({'column_name': table_a.columns, 'data_example': table_a.iloc[0].values})
    columns_b = pd.DataFrame({'column_name': table_b.columns, 'data_example': table_b.iloc[0].values})

    if verbose:
        print('Table A columns:')
        print(columns_a[1:], '\n')
        print('Table B columns:')
        print(columns_b[1:], '\n')

    if cols_a_to_rm:
        table_a = table_a.drop(columns=cols_a_to_rm, errors='ignore')
        print('Removed columns from A:', cols_a_to_rm) if verbose else None
    if cols_b_to_rm:
        table_b = table_b.drop(columns=cols_b_to_rm, errors='ignore')
        print('Removed columns from B:', cols_b_to_rm) if verbose else None

    # Check if the columns contains the same values, they could have different length
    if verbose and table_a.columns.values.tolist() != table_b.columns.values.tolist():
        print('Columns are not the same in both tables')
        print('Table A columns:', table_a.columns.values)
        print('Table B columns:', table_b.columns.values)
        if set(table_a.columns.values) == set(table_b.columns.values):
            print('Columns are the same but in different order')

    if order_cols:
        # Get the max length by values and col name and re order to have the small value cols at the beginning
        if verbose:
            print('Reordering columns')
            print('Table A columns order before:', table_a.columns.values)
            print('Table B columns order before:', table_b.columns.values)
        table_a = table_a.reindex(sorted(table_a.columns, key=lambda x: (table_a[x].map(str).apply(len).max(), x)),
                                  axis=1)
        if set(table_a.columns.values) == set(table_b.columns.values):
            table_b = table_b[table_a.columns]
        else:
            table_b = table_b.reindex(sorted(table_b.columns, key=lambda x: (table_b[x].map(str).apply(len).max(), x)),
                                      axis=1)
        if verbose:
            print('Table A columns order after:', table_a.columns.values)
            print('Table B columns order after:', table_b.columns.values)

    table_a_serialized = [serialize_entities(entity, remove_col_names=remove_col_names) for (_, entity) in
                          table_a.iterrows()]
    table_b_serialized = [serialize_entities(entity, remove_col_names=remove_col_names) for (_, entity) in
                          table_b.iterrows()]
    if verbose:
        print('Serialized entities', '\n')

    X_train, y_train, X_valid, y_valid, X_test, y_test = [], [], [], [], [], []

    if pairs_train is not None:
        X_train, y_train = [(pairs_train['source_id'].values[i], pairs_train['target_id'].values[i]) for i in
                            range(len(pairs_train))], [1 if pairs_train['matching'].values[i] else 0 for i in
                                                       range(len(pairs_train))]
    if pairs_val is not None:
        X_valid, y_valid = [(pairs_val['source_id'].values[i], pairs_val['target_id'].values[i]) for i in
                            range(len(pairs_val))], [1 if pairs_val['matching'].values[i] else 0 for i in
                                                     range(len(pairs_val))]
    if pairs_test is not None:
        X_test, y_test = [(pairs_test['source_id'].values[i], pairs_test['target_id'].values[i]) for i in
                          range(len(pairs_test))], [1 if pairs_test['matching'].values[i] else 0 for i in
                                                    range(len(pairs_test))]

    if return_tables:
        return table_a_serialized, table_b_serialized, table_a, table_b, X_train, y_train, X_valid, y_valid, X_test, y_test
    else:
        return table_a_serialized, table_b_serialized, X_train, y_train, X_valid, y_valid, X_test, y_test


def add_transitive(preds, X_test_pairs):
    """
    Add transitive closure to the predictions.

    This function takes the predictions of a model and adds transitive closure to them.
    If the model predicts that A matches B and C matches with B & D then A matches D.

    Parameters:
    -----------
    preds : np.ndarray
        The model's predictions.
    X_test_pairs : np.ndarray
        The pairs of entities used for testing.

    Returns:
    --------
    np.ndarray
        The predictions with transitive closure added.
    """

    # TODO: Adapt it in the case of the full pipeline

    # Create a dictionary to map pairs to their predictions
    pair_pred = {(e1, e2): pred for (e1, e2), pred in zip(X_test_pairs, preds)}
    pair_pred_copy = pair_pred.copy()

    # Add transitive closure to the predictions
    for (e1, e2) in pair_pred:
        for (e3, e4) in pair_pred:
            if pair_pred[(e1, e2)] and pair_pred[(e3, e4)]:
                if (e3, e2) in pair_pred and pair_pred[(e3, e2)] and (e1, e4) in pair_pred and not pair_pred[(e1, e4)]:
                    pair_pred[(e1, e4)] = 1
                    print(f'Transitive closure: {e1} matches {e4}')
                if (e1, e4) in pair_pred and pair_pred[(e1, e4)] and (e3, e2) in pair_pred and not pair_pred[(e3, e2)]:
                    pair_pred[(e3, e2)] = 1
                    print(f'Transitive closure: {e3} matches {e2}')

    # Convert the dictionary back to a list of predictions
    return np.array([pair_pred[(e1, e2)] for e1, e2 in X_test_pairs])


###################################################### OLD  ######################################################

def split_dataset_with_neg_sampling(pairs, neg_sample_ratio=1, train_ratio=0.7, val_ratio=0.15, seed=None):
    """
    Splits a dataset of positive pairs into training, validation, and test sets with negative sampling.

    :param pairs: list of tuples
        A list of tuples where each tuple represents a positive pair (idA, idB).
    :param neg_sample_ratio: int, optional
        The ratio of negative samples to positive samples. Default is 1.
    :param train_ratio: float, optional
        The ratio of the dataset to be used for training. Default is 0.7.
    :param val_ratio: float, optional
        The ratio of the dataset to be used for validation. Default is 0.15.
    :param seed: int, optional
        Random seed for reproducibility. Default is None.

    :returns: tuple
        - **X_train** (*numpy.ndarray*): Array of training pairs (both positive and negative).
        - **y_train** (*numpy.ndarray*): Array of labels for the training pairs (1 for positive, 0 for negative).
        - **X_val** (*numpy.ndarray*): Array of validation pairs (both positive and negative).
        - **y_val** (*numpy.ndarray*): Array of labels for the validation pairs (1 for positive, 0 for negative).
        - **X_test** (*numpy.ndarray*): Array of test pairs (both positive and negative).
        - **y_test** (*numpy.ndarray*): Array of labels for the test pairs (1 for positive, 0 for negative).

    :note:
        - The function first builds a graph from the positive pairs and finds connected components.
        - The components are then shuffled and split into training, validation, and test sets.
        - Negative samples are generated by randomly pairing nodes that are not connected in the original graph.
        - The function returns the concatenated positive and negative samples along with their labels.
    """
    if seed is not None:
        random.seed(seed)

    # Step 1: Build a graph where idA and idB are nodes, and each pair is an edge
    graph = defaultdict(set)

    for idA, idB in pairs:
        graph[idA].add(idB)
        graph[idB].add(idA)

    # Step 2: Find connected components
    def find_component(node, visited, component):
        visited.add(node)
        component.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                find_component(neighbor, visited, component)

    components = []
    visited = set()

    for node in graph:
        if node not in visited:
            component = set()
            find_component(node, visited, component)
            components.append(component)

    # Step 3: Shuffle the components to ensure random split
    random.shuffle(components)

    # Step 4: Split components into train, validation, and test sets
    num_components = len(components)

    train_size = int(train_ratio * num_components)
    val_size = int(val_ratio * num_components)

    train_components = components[:train_size]
    val_components = components[train_size:train_size + val_size]
    test_components = components[train_size + val_size:]

    # Step 5: Extract pairs from the components
    def extract_pairs(components):
        pairs_set = set()
        for component in components:
            for node in component:
                for neighbor in graph[node]:
                    if (node, neighbor) in pairs or (neighbor, node) in pairs:
                        if node[0] == 'a':
                            pairs_set.add((node, neighbor))
                        else:
                            pairs_set.add((neighbor, node))
        return pairs_set

    train_set = extract_pairs(train_components)
    val_set = extract_pairs(val_components)
    test_set = extract_pairs(test_components)

    pairs_train = list(train_set)
    pairs_val = list(val_set)
    pairs_test = list(test_set)

    # Step 6: Generate negative samples

    def get_neg_sample(pairs_list):
        neg_sample = []
        for _ in range(neg_sample_ratio * len(pairs_list)):
            e1 = random.choice(pairs_list)[0]
            e2 = random.choice(pairs_list)[1]
            while (e1, e2) in pairs_list:
                e2 = random.choice(pairs_list)[1]
            neg_sample.append((e1, e2))
        return neg_sample

    neg_pairs_train = get_neg_sample(pairs_train)
    neg_pairs_val = get_neg_sample(pairs_val)
    neg_pairs_test = get_neg_sample(pairs_test)

    # Step 7: Concatenate positive and negative samples
    X_train = np.array(pairs_train + neg_pairs_train)
    X_val = np.array(pairs_val + neg_pairs_val)
    X_test = np.array(pairs_test + neg_pairs_test)
    y_train = np.array([1] * len(pairs_train) + [0] * len(neg_pairs_train))
    y_val = np.array([1] * len(pairs_val) + [0] * len(neg_pairs_val))
    y_test = np.array([1] * len(pairs_test) + [0] * len(neg_pairs_test))

    # Step 8: Shuffle the data
    indices_train = np.arange(len(X_train))
    indices_val = np.arange(len(X_val))
    indices_test = np.arange(len(X_test))

    random.shuffle(indices_train)
    random.shuffle(indices_val)
    random.shuffle(indices_test)

    X_train = X_train[indices_train]
    y_train = y_train[indices_train]
    X_val = X_val[indices_val]
    y_val = y_val[indices_val]
    X_test = X_test[indices_test]
    y_test = y_test[indices_test]

    return X_train, y_train, X_val, y_val, X_test, y_test

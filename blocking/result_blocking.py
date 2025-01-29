## Import
import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

while 'model' not in os.listdir():
    os.chdir('..')

import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn
from sentence_transformers import (CrossEncoder, InputExample,
                                   SentencesDataset, SentenceTransformer)
from sentence_transformers.cross_encoder.evaluation import \
    CEBinaryClassificationEvaluator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.neighbors import NearestNeighbors
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.notebook import tqdm, trange

from model.Blocking import (get_blocking_metrics, merge_indices,
                            merge_true_matches, perform_blocking_sbert,
                            perform_blocking_tfidf)
from model.utils import add_transitive, load_data

## Constants

DATA_NAMES = ['fodors-zagats', 'amazon-google', 'abt-buy']
LOAD_OPTIONS = {
    'order_cols' : [True, False],
    'remove_col_names' : [True, False],
}
MODELS = [
    'sentence-transformers/allenai-specter',
    'all-distilroberta-v1',
    'all-mpnet-base-v2',
    'multi-qa-mpnet-base-dot-v1',
]

## Hyperparameters

EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
CLASSIFIER_DROPOUT = None
threshold = 0.65
K = 15

## Functions

def load_results(data_name, model_name, order_cols, remove_col_names):
    dir_name = f'{data_name}-blocking/{model_name}-order_cols_{order_cols}-remove_col_names_{remove_col_names}'
    with open(f'results/{dir_name}/train_time.txt', 'r') as f:
        train_time = float(f.read())
    with open(f'results/{dir_name}/pairs.txt', 'r') as f:
        neighbors = [line.strip().split(',') for line in f.readlines()]
    return train_time, neighbors

## Main

if __name__ == '__main__':
    
    ## Make directories

    for data_name in tqdm(DATA_NAMES, desc='Datasets', position=0):
        for model_name in tqdm(MODELS, desc='Models', position=1, leave=False):
            for order_cols in tqdm(LOAD_OPTIONS['order_cols'], desc='Order columns', position=2, leave=False):
                for remove_col_names in tqdm(LOAD_OPTIONS['remove_col_names'], desc='Remove column names', position=3, leave=False):
                    dir_name = f'{data_name}-blocking/{model_name}-order_cols_{order_cols}-remove_col_names_{remove_col_names}'
                    os.makedirs(os.path.join('results', dir_name), exist_ok=True)


    ## Test

    for dataset in tqdm(DATA_NAMES, desc='Datasets', position=0):
        for model_name in tqdm(MODELS, desc='Models', position=1, leave=False):
            for order_cols in tqdm(LOAD_OPTIONS['order_cols'], desc='Order columns', position=2, leave=False):
                for remove_col_names in tqdm(LOAD_OPTIONS['remove_col_names'], desc='Remove column names', position=3, leave=False):
                    print(f'Dataset: {dataset}, Model: {model_name}, Order columns: {order_cols}, Remove column names: {remove_col_names}')
                    data_name = dataset
                    data_dir = os.path.join('data', data_name)

                    table_a_serialized, table_b_serialized, X_train_ids, y_train, X_valid_ids, y_valid, X_test_ids, y_test = load_data(data_dir, order_cols=order_cols, remove_col_names=remove_col_names)

                    all_true_matches = merge_true_matches(X_train_ids, y_train, X_valid_ids, y_valid, X_test_ids, y_test)
                    
                    start_time = time.time()
                    blocked_pairs = perform_blocking_sbert(model_name, table_a_serialized, table_b_serialized, n_neighbors=K, metric='cosine', device=device)
                    blocked_pairs = merge_indices(blocked_pairs,
                                              perform_blocking_tfidf(table_a_serialized, table_b_serialized,
                                                                     n_neighbors=K, metric='cosine'))
                    training_time = time.time() - start_time

                    print(f'Training time: {training_time}')
                    
                    # Save train time and logits to file
                    with open(f'results/{data_name}-blocking/{model_name}-order_cols_{order_cols}-remove_col_names_{remove_col_names}/train_time.txt', 'w') as f:
                        f.write(str(training_time))
                    
                    with open(f'results/{data_name}-blocking/{model_name}-order_cols_{order_cols}-remove_col_names_{remove_col_names}/pairs.txt', 'w') as f:
                        for neighbor in blocked_pairs:
                            f.write(','.join(map(str, neighbor)) + '\n')
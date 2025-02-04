## Import
import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

while 'model' not in os.listdir():
    os.chdir('..')

import time

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from model.utils import load_data

## Constants

DATA_NAMES = ['fodors-zagats', 'amazon-google', 'abt-buy']
LOAD_OPTIONS = {
    'order_cols': [True, False],
    'remove_col_names': [True, False],
}
MODELS = ['all-mpnet-base-v2', 'sentence-transformers/multi-qa-mpnet-base-dot-v1']

## Hyperparameters

EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
CLASSIFIER_DROPOUT = None
threshold = 0.65


## Functions

def load_results(data_name, model_name, order_cols, remove_col_names):
    dir_name = f'{data_name}-zshot/{model_name}-order_cols_{order_cols}-remove_col_names_{remove_col_names}'
    with open(f'results/{dir_name}/train_time.txt', 'r') as f:
        train_time = float(f.read())
    with open(f'results/{dir_name}/logits.csv', 'r') as f:
        logits = [line.strip().split(',') for line in f.readlines()]
    return train_time, logits


## Main

if __name__ == '__main__':

    ## Make directories

    for data_name in tqdm(DATA_NAMES, desc='Datasets', position=0):
        for model_name in tqdm(MODELS, desc='Models', position=1, leave=False):
            for order_cols in tqdm(LOAD_OPTIONS['order_cols'], desc='Order columns', position=2, leave=False):
                for remove_col_names in tqdm(LOAD_OPTIONS['remove_col_names'], desc='Remove column names', position=3,
                                             leave=False):
                    dir_name = f'{data_name}-zshot/{model_name}-order_cols_{order_cols}-remove_col_names_{remove_col_names}'
                    os.makedirs(os.path.join('results', dir_name), exist_ok=True)

    ## Test

    for dataset in tqdm(DATA_NAMES, desc='Datasets', position=0):
        for model_name in tqdm(MODELS, desc='Models', position=1, leave=False):
            for order_cols in tqdm(LOAD_OPTIONS['order_cols'], desc='Order columns', position=2, leave=False):
                for remove_col_names in tqdm(LOAD_OPTIONS['remove_col_names'], desc='Remove column names', position=3,
                                             leave=False):
                    print(
                        f'Dataset: {dataset}, Model: {model_name}, Order columns: {order_cols}, Remove column names: {remove_col_names}')
                    data_name = dataset
                    data_dir = os.path.join('data', data_name)

                    table_a_serialized, table_b_serialized, X_train_ids, y_train, X_valid_ids, y_valid, X_test_ids, y_test = load_data(
                        data_dir, order_cols=order_cols, remove_col_names=remove_col_names)

                    X1_train, X2_train = [table_a_serialized[i[0]] for i in X_train_ids], [table_b_serialized[i[1]] for
                                                                                           i in X_train_ids]
                    X1_valid, X2_valid = [table_a_serialized[i[0]] for i in X_valid_ids], [table_b_serialized[i[1]] for
                                                                                           i in X_valid_ids]
                    X1_test, X2_test = [table_a_serialized[i[0]] for i in X_test_ids], [table_b_serialized[i[1]] for i
                                                                                        in X_test_ids]

                    model = SentenceTransformer(model_name, device=device)

                    valid_evaluation_set = [(e1, e2) for e1, e2 in zip(X1_valid, X2_valid)]

                    start_time = time.time()
                    embedding1_test = model.encode(X1_test, show_progress_bar=True)
                    embedding2_test = model.encode(X2_test, show_progress_bar=True)
                    training_time = time.time() - start_time

                    print(f'Training time: {training_time}')

                    similarity_test_zero = cosine_similarity(embedding1_test, embedding2_test)

                    logits = [similarity_test_zero[i, i] for i in range(len(similarity_test_zero))]

                    print(classification_report(y_test, np.array(logits) > threshold))

                    # Save train time and logits to file
                    with open(
                            f'results/{data_name}-zshot/{model_name}-order_cols_{order_cols}-remove_col_names_{remove_col_names}/train_time.txt',
                            'w') as f:
                        f.write(str(training_time))

                    with open(
                            f'results/{data_name}-zshot/{model_name}-order_cols_{order_cols}-remove_col_names_{remove_col_names}/logits.csv',
                            'w') as f:
                        for i, logit in enumerate(logits):
                            f.write(f'{X_test_ids[i][0]},{X_test_ids[i][1]},{logit}\n')

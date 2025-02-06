## Import
import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

while 'model' not in os.listdir():
    os.chdir('..')

import time

import torch.nn as nn
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import \
    CEBinaryClassificationEvaluator
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.utils import load_data

## Constants

DATA_NAMES = ['fodors-zagats', 'amazon-google', 'abt-buy']
LOAD_OPTIONS = {
    'order_cols': [True, False],
    'remove_col_names': [True, False],
}
MODELS = ['cross-encoder/stsb-roberta-base', 'cross-encoder/stsb-distilroberta-base',
          'cross-encoder/ms-marco-MiniLM-L-12-v2', 'cross-encoder/stsb-roberta-large']

## Hyperparameters

EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
CLASSIFIER_DROPOUT = None
threshold = 0.65


## Functions

def load_results(data_name, model_name, order_cols, remove_col_names):
    dir_name = f'{data_name}-{model_name}-order_cols_{order_cols}-remove_col_names_{remove_col_names}'
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
                    dir_name = f'{data_name}-{model_name}-order_cols_{order_cols}-remove_col_names_{remove_col_names}'
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
                    
                    # If results already exist, skip
                    if os.path.exists(
                            f'results/{data_name}-{model_name}-order_cols_{order_cols}-remove_col_names_{remove_col_names}/logits.csv'):
                        continue

                    table_a_serialized, table_b_serialized, X_train_ids, y_train, X_valid_ids, y_valid, X_test_ids, y_test = load_data(
                        data_dir, order_cols=order_cols, remove_col_names=remove_col_names)

                    X1_train, X2_train = [table_a_serialized[i[0]] for i in X_train_ids], [table_b_serialized[i[1]] for
                                                                                           i in X_train_ids]
                    X1_valid, X2_valid = [table_a_serialized[i[0]] for i in X_valid_ids], [table_b_serialized[i[1]] for
                                                                                           i in X_valid_ids]
                    X1_test, X2_test = [table_a_serialized[i[0]] for i in X_test_ids], [table_b_serialized[i[1]] for i
                                                                                        in X_test_ids]

                    train_datasets = [InputExample(texts=[X1_train[i], X2_train[i]], label=y_train[i]) for i in
                                      range(len(X_train_ids))]
                    valid_datasets = [InputExample(texts=[X1_valid[i], X2_valid[i]], label=y_valid[i]) for i in
                                      range(len(X_valid_ids))]
                    test_datasets = [InputExample(texts=[X1_test[i], X2_test[i]], label=y_test[i]) for i in
                                     range(len(X_test_ids))]
                    train_loader = DataLoader(train_datasets, shuffle=True, batch_size=16, num_workers=0)
                    valid_loader = DataLoader(valid_datasets, shuffle=False, batch_size=16, num_workers=0)
                    test_examples = DataLoader(test_datasets, shuffle=False, batch_size=16, num_workers=0)

                    valid_evaluation_set = [(e1, e2) for e1, e2 in zip(X1_valid, X2_valid)]

                    loss_fct = BCEWithLogitsLoss()

                    model = CrossEncoder(model_name, num_labels=1, device=device, classifier_dropout=CLASSIFIER_DROPOUT,
                                         default_activation_function=nn.Sigmoid())

                    start_time = time.time()
                    model.fit(train_dataloader=train_loader,
                              loss_fct=loss_fct,
                              evaluator=CEBinaryClassificationEvaluator(valid_evaluation_set, labels=y_valid,
                                                                        show_progress_bar=False),
                              epochs=EPOCHS,
                              warmup_steps=WARMUP_STEPS,
                              optimizer_params={'lr': LR},
                              weight_decay=WEIGHT_DECAY,
                              show_progress_bar=False)
                    training_time = time.time() - start_time

                    print(f'Training time: {training_time}')

                    logits = model.predict([[e1, e2] for e1, e2 in zip(X1_test, X2_test)], apply_softmax=True)

                    # Save train time and logits to file
                    with open(
                            f'results/{data_name}-{model_name}-order_cols_{order_cols}-remove_col_names_{remove_col_names}/train_time.txt',
                            'w') as f:
                        f.write(str(training_time))

                    with open(
                            f'results/{data_name}-{model_name}-order_cols_{order_cols}-remove_col_names_{remove_col_names}/logits.csv',
                            'w') as f:
                        for i, logit in enumerate(logits):
                            f.write(f'{X_test_ids[i][0]},{X_test_ids[i][1]},{logit}\n')

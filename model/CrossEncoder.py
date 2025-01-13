import time

import torch.nn as nn
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import \
    CEBinaryClassificationEvaluator
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from model.utils import load_data


def prepare_data_cross_encoder(data_dir, remove_col_names=True, order_cols=True, blocked_pairs=None):
    table_a_serialized, table_b_serialized, X_train_ids, y_train, X_valid_ids, y_valid, X_test_ids, y_test = load_data(
        data_dir, remove_col_names=remove_col_names, order_cols=order_cols)

    if blocked_pairs is not None:
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

        all_matches = []
        for i in range(len(table_a_serialized)):
            for j in blocked_pairs[i]:
                all_matches.append((i, j))
        X_train_ids, X_test_ids, y_train, y_test = train_test_split(all_matches,
                                                                    [1 if x in all_true_matches else 0 for x in
                                                                     all_matches], test_size=0.2, random_state=42)
        print('Blocked pairs used')

    X1_train, X2_train = [table_a_serialized[i[0]] for i in X_train_ids], [table_b_serialized[i[1]] for i in
                                                                           X_train_ids]
    X1_valid, X2_valid = [table_a_serialized[i[0]] for i in X_valid_ids], [table_b_serialized[i[1]] for i in
                                                                           X_valid_ids]
    X1_test, X2_test = [table_a_serialized[i[0]] for i in X_test_ids], [table_b_serialized[i[1]] for i in X_test_ids]

    train_datasets = [InputExample(texts=[X1_train[i], X2_train[i]], label=y_train[i]) for i in range(len(X_train_ids))]
    train_loader = DataLoader(train_datasets, shuffle=True, batch_size=16, num_workers=0)

    test_set = [(e1, e2) for e1, e2 in zip(X1_test, X2_test)]
    valid_set = [(e1, e2) for e1, e2 in zip(X1_valid, X2_valid)]

    return train_loader, valid_set, y_valid, test_set, y_test


def fit_cross_encoder(model_name, train_loader, valid_set, y_valid, test_set, epochs=1, learning_rate=2e-5,
                      weight_decay=0.01, classifier_dropout=None, device='cpu'):
    loss_fct = BCEWithLogitsLoss()

    model = CrossEncoder(model_name, num_labels=1, device=device, classifier_dropout=classifier_dropout,
                         default_activation_function=nn.Sigmoid())

    warmup_steps = int(len(train_loader) * epochs * 0.1)

    start_train_time = time.time()
    model.fit(train_dataloader=train_loader,
              loss_fct=loss_fct,
              evaluator=CEBinaryClassificationEvaluator(valid_set, labels=y_valid, show_progress_bar=True),
              epochs=epochs,
              warmup_steps=warmup_steps,
              callback=lambda epoch, steps, loss: print(f'Epoch: {epoch}, Steps: {steps}, Loss: {loss}'),
              optimizer_params={'lr': learning_rate},
              weight_decay=weight_decay,
              show_progress_bar=True)
    train_time = time.time() - start_train_time

    logits = model.predict(test_set, apply_softmax=True)

    return logits, train_time


def evaluate_cross_encoder(logits, y_test, threshold=0.65):
    predictions = (logits > threshold).astype(int)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, logits)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 score: {f1}')
    print(f'ROC AUC: {roc_auc}')

    return accuracy, precision, recall, f1, roc_auc

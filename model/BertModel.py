#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File name: BertModel.py
Author: Tristan PERROT
Created: 23/09/2024
Version: 1.0
Description: PyTorch model for entity resolution using BERT
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ERBertDataset(Dataset):
    """
    A custom Dataset class for entity resolution using BERT.

    Attributes
    ----------
    X : list or array-like
        The input data (serialized entities).
    y : list or array-like
        The labels corresponding to the input data.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer for preprocessing the input data.
    max_length : int
        The maximum sequence length for tokenization.

    Methods
    -------
    __len__():
        Returns the number of samples in the dataset.
    __getitem__(idx):
        Returns the tokenized input and label for the given index.
    """

    def __init__(self, X, y, tokenizer):
        """
        Initializes the ERBertDataset.

        Args:
            X (list or array-like): The input data (serialized entities).
            y (list or array-like): The labels corresponding to the input data.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer for preprocessing.
        """
        self.X = X
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieves the tokenized input and label for the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing input_ids, attention_mask, and labels.
        """
        text = self.X[idx]
        label = self.y[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Squeeze to remove the batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item


class BertModel(nn.Module):
    """
    A PyTorch model for text classification using a pre-trained BERT model.

    Attributes
    ----------
    device : str
        The device to run the model on (e.g., "cuda" or "cpu").
    bert : transformers.PreTrainedModel
        The pre-trained BERT model.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer for the pre-trained BERT model.
    writer : torch.utils.tensorboard.SummaryWriter
        The TensorBoard writer for logging.
    fc : torch.nn.Linear
        The fully connected layer for classification.

    Methods
    -------
    __init__(model_name="roberta-base", device="cuda")
        Initializes the BertModel with a specified pre-trained model and device.
    forward(input_ids, attention_mask)
        Performs a forward pass through the model.
    train_step(X_train, y_train, optimizer, loss_fn)
        Performs a single training step.
    eval_step(X_val, y_val, loss_fn)
        Performs a single evaluation step.
    predict(X_test)
        Generates predictions for the given test data.
    fit(X_train, y_train, X_val, y_val, epochs=10,
        batch_size=32, lr=2e-5, early_stopping=True, patience=3)
        Trains the model on the training data and evaluates it on the validation data.
    evaluate(X_test, y_test)
        Evaluates the model on the test data and prints a classification report.
    save(path)
        Saves the model's state dictionary to the specified path.
    load(path)
        Loads the model's state dictionary from the specified path.
    """

    def __init__(self, model_name="roberta-base", study_name="Unknown", device="cpu"):
        """
        Initializes the BertModel with a specified pre-trained model and device.

        :param model_name: The name of the pre-trained model to use, defaults to "roberta-base".
        :type model_name: str, optional
        :param study_name: The name of the study for TensorBoard logging, defaults to "Unknown".
        :type study_name: str, optional
        :param device: The device to run the model on, defaults to "cpu".
        :type device: str, optional
        """
        super().__init__()

        self.device = device
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True, sep_token='[SEP]')
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[COL]', '[VAL]']})
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.writer = SummaryWriter("runs/" + study_name)

        self.model_name = model_name

        print("Model initialized")
        print(f"Device: {self.device}")
        print(f"Model: {model_name}")
        print(f"Study name: {study_name}")

    def forward(self, input_ids, attention_mask):
        """
        Perform a forward pass through the BERT model.
        Args:
            input_ids (torch.Tensor): Tensor of input token IDs.
            attention_mask (torch.Tensor): Tensor of attention masks to avoid attending to padding tokens.
        Returns:
            torch.Tensor: Logits obtained from the final fully connected layer.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits
    
    def prepare_data(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32, num_workers=4):
        """
        Prepares the data for training, validation, and testing.

        Args:
            X_train (list or array-like): Training input data.
            y_train (list or array-like): Training labels.
            X_val (list or array-like): Validation input data.
            y_val (list or array-like): Validation labels.
            X_test (list or array-like): Test input data.
            y_test (list or array-like): Test labels.
            batch_size (int, optional): Batch size, defaults to 32.
            num_workers (int, optional): Number of workers for data loading, defaults to 4.

        Returns:
            tuple: Training, validation, and test DataLoader objects.
        """
        train_dataset = ERBertDataset(X_train, y_train, self.tokenizer)
        val_dataset = ERBertDataset(X_val, y_val, self.tokenizer)
        test_dataset = ERBertDataset(X_test, y_test, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

        return train_loader, val_loader, test_loader

    def fit(self, train_loader, val_loader, epochs=10, lr=2e-5, weight_decay=0.01, early_stopping=True, patience=3):
        """
        Trains the model on the training data and evaluates it on the validation data.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            epochs (int, optional): Number of epochs to train the model, defaults to 10.
            lr (float, optional): Learning rate for the optimizer, defaults to 2e-5.
            weight_decay (float, optional): Weight decay for the optimizer, defaults to 0.01.
            early_stopping (bool, optional): Whether to use early stopping, defaults to True.
            patience (int, optional): Number of epochs to wait before early stopping, defaults to 3.

        Returns:
            None
        """
        self.to(self.device)
        self.train()

        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.BCELoss()

        best_val_loss = float('inf')
        best_model_weights = None
        patience_counter = 0

        self.time_per_epoch = []

        for epoch in range(epochs):
            start_time = time.time()
            print(f"Epoch {epoch+1}/{epochs}")

            # Training Loop
            self.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc="Training", unit="batch"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                labels_onehot = F.one_hot(labels, num_classes=2).float()

                logits = self(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                loss = loss_fn(probs, labels_onehot)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation Loop
            self.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", unit="batch"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    labels_onehot = F.one_hot(labels, num_classes=2).float()

                    logits = self(input_ids, attention_mask)
                    probs = torch.softmax(logits, dim=1)
                    loss = loss_fn(probs, labels_onehot)
                    val_loss += loss.item()

                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            class_report = classification_report(all_labels, all_preds)

            print(f"Train loss: {avg_train_loss:.4f}")
            print(f"Val loss: {avg_val_loss:.4f}")
            print(class_report)

            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/val', avg_val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', (np.array(all_preds) == np.array(all_labels)).mean(), epoch)
            self.writer.add_text('Classification Report Validation', class_report, epoch)

            # Early Stopping Check
            if early_stopping:
                self.writer.add_scalar('Patience', patience_counter, epoch)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_weights = self.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered")
                        if best_model_weights is not None:
                            self.load_state_dict(best_model_weights)
                            print("Best weights restored")

                        mean_time = sum(self.time_per_epoch) / len(self.time_per_epoch) if len(self.time_per_epoch) > 0 else 0
                        print(f"Mean time per epoch: {mean_time:.2f}")
                        return

            duration = time.time() - start_time
            self.time_per_epoch.append(duration)

        print("Training completed")
        if best_model_weights is not None:
            self.load_state_dict(best_model_weights)
            print("Best weights restored")

        mean_time = sum(self.time_per_epoch) / len(self.time_per_epoch) if len(self.time_per_epoch) > 0 else 0
        print(f"Mean time per epoch: {mean_time:.2f}")

    def evaluate(self, test_loader):
        """
        Evaluates the model on the test data and prints a classification report.

        Args:
            test_loader (DataLoader): DataLoader for test data.

        Returns:
            torch.Tensor: Predictions for the test data.
        """
        self.eval()
        loss_fn = nn.BCELoss()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", unit="batch"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                labels_onehot = F.one_hot(labels, num_classes=2).float()

                logits = self(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                loss = loss_fn(probs, labels_onehot)
                test_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        class_report = classification_report(all_labels, all_preds)

        print(class_report)
        print(f"Test loss: {avg_test_loss:.5f}")

        self.writer.add_text('Classification Report Test', class_report)
        self.writer.add_scalar('Accuracy/test', (np.array(all_preds) == np.array(all_labels)).mean())
        self.writer.add_scalar('Loss/test', avg_test_loss)

        return torch.tensor(all_preds, device=self.device)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def predict_entities(self, entitie1, entitie2):
        """
        Predicts whether two entities match based on their serialized representation.
        Args:
            entitie1 (pd.Series): The first entity to compare.
            entitie2 (pd.Series): The second entity to compare.
        Returns:
            int: The predicted label (0 for no match, 1 for match).
        Notes:
            - The two entities are serialized into a single string.
            - The serialized string is tokenized and converted to input IDs and attention mask.
            - The model predicts whether the two entities match.
        """
        input_string = serialize_entities(entitie1, entitie2)
        input_ids = self.tokenizer(
            input_string, return_tensors="pt")["input_ids"].to(self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            logits = self(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

        return preds.item()

    def predict_entities_proba(self, entitie1, entitie2):
        """
        Predicts the probability that two entities match based on their serialized representation.
        Args:
            entitie1 (pd.Series): The first entity to compare.
            entitie2 (pd.Series): The second entity to compare.
        Returns:
            float: The probability that the two entities match.
        Notes:
            - The two entities are serialized into a single string.
            - The serialized string is tokenized and converted to input IDs and attention mask.
            - The model predicts the probability that the two entities match.
        """
        input_string = serialize_entities(entitie1, entitie2)
        input_ids = self.tokenizer(input_string, return_tensors="pt")["input_ids"].to(self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            logits = self(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)

        return probs[0, 1].item()

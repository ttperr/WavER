# Results of the different models

As of now, the results are provided running on a **NVIDIA A100-40GiB GPU**. The **batch size is set to 32** and the learning rate is set to 2e-5. The models are trained for **10 epochs**. There is an **early stopping** mechanism that stops the training if the validation loss does not decrease for 3 epochs. Otherwise, the model is trained on **10 epochs**.

## [BERT Model](../model/BertModel.py)

### Abt-Buy - Test Set - BERT Model

| Model         | Accuracy | Precision | Recall | F1   | Loss    | Time/epoch |
| ------------- | -------- | --------- | ------ | ---- | ------- | ---------- |
| RoBERTa Base  | 1.00     | 0.99      | 1.00   | 1.00 | 0.01014 | 33s        |
| DistilRoBERTa | 0.98     | 0.99      | 0.98   | 0.98 | 0.02789 | **17s**    |
| BERT Base     | 1.00     | 0.99      | 1.00   | 1.00 | 0.01531 | 33s        |
| Electra Base  | 0.99     | 0.99      | 1.00   | 0.99 | 0.01888 | 19s        |

### Amazon-Google - Test Set - BERT Model

| Model         | Accuracy | Precision | Recall | F1   | Loss    | Time/epoch |
| ------------- | -------- | --------- | ------ | ---- | ------- | ---------- |
| RoBERTa Base  | 0.95     | 0.93      | 0.98   | 0.95 | 0.08975 | 39s        |
| DistilRoBERTa | 0.95     | 0.97      | 0.92   | 0.95 | 0.12640 | **20s**    |
| BERT Base     | 0.95     | 0.93      | 0.97   | 0.95 | 0.07849 | 39s        |
| Electra Base  | 0.96     | 0.96      | 0.95   | 0.96 | 0.09199 | 45s        |

### Fodors-Zagats - Test Set - BERT Model

| Model         | Accuracy | Precision | Recall | F1   | Loss        | Time/epoch |
| ------------- | -------- | --------- | ------ | ---- | ----------- | ---------- |
| RoBERTa Base  | 1.00     | 1.00      | 1.00   | 1.00 | **0.00093** | 3.79s      |
| DistilRoBERTa | 1.00     | 1.00      | 1.00   | 1.00 | 0.02167     | **2.25s**  |
| BERT Base     | 1.00     | 1.00      | 1.00   | 1.00 | 0.00136     | 3.96s      |
| Electra Base  | 1.00     | 1.00      | 1.00   | 1.00 | 0.01513     | 4.39s      |

### Walmart-Amazon - Test Set - BERT Model

| Model         | Accuracy | Precision | Recall | F1   | Loss    | Time/epoch |
| ------------- | -------- | --------- | ------ | ---- | ------- | ---------- |
| RoBERTa Base  | 0.85     | 0.84      | 0.87   | 0.85 | 0.22031 | 36s        |
| DistilRoBERTa | 0.85     | 0.81      | 0.92   | 0.86 | 0.21455 | **18s**    |
| BERT Base     | 0.87     | 0.81      | 0.98   | 0.88 | 0.21891 | 36s        |
| Electra Base  | 0.85     | 0.96      | 0.73   | 0.83 | 0.25164 | 41s        |

## [Cross Encoder SBERT Model](../code-sberts/)

Now, there is no early stopping mechanism for the SBERT models. The models are trained for **3 epochs**.

### Abt-Buy - Test Set - SBERT Model

| Model                  | Accuracy | Precision | Recall | F1   | Loss    | Time/epoch |
| ---------------------- | -------- | --------- | ------ | ---- | ------- | ---------- |
| MS Marco MiniLM l-12   | 0.99     | 0.99      | 1.00   | 0.99 | 0.01581 | 6s         |
| STS RoBERTa-base       | 0.98     | 1.00      | 0.96   | 0.98 | 0.51097 | 17s        |
| STS DistilRoBERTa-base | 1.00     | 1.00      | 1.00   | 1.00 | 0.51683 | 9s         |

### Amazon-Google - Test Set - SBERT Model

| Model                  | Accuracy | Precision | Recall | F1   | Loss    | Time/epoch |
| ---------------------- | -------- | --------- | ------ | ---- | ------- | ---------- |
| MS Marco MiniLM l-12   | 0.97     | 0.95      | 0.99   | 0.97 | 0.08571 | 15s        |
| STS RoBERTa-base       | 0.99     | 0.99      | 1.00   | 0.99 | 0.50828 | 40s        |
| STS DistilRoBERTa-base | 0.99     | 0.99      | 0.99   | 0.99 | 0.50684 | 20s        |

### Fodors-Zagats - Test Set - SBERT Model

| Model                  | Accuracy | Precision | Recall | F1   | Loss    | Time/epoch |
| ---------------------- | -------- | --------- | ------ | ---- | ------- | ---------- |
| MS Marco MiniLM l-12   | 0.75     | 0.67      | 1.00   | 0.80 | 0.85433 | 2s         |
| STS RoBERTa-base       | 1.00     | 1.00      | 1.00   | 1.00 | 0.53191 | 1s         |
| STS DistilRoBERTa-base | 1.00     | 1.00      | 1.00   | 1.00 | 0.54098 | 1s         |

### Walmart-Amazon - Test Set - SBERT Model

| Model                  | Accuracy | Precision | Recall | F1   | Loss    | Time/epoch |
| ---------------------- | -------- | --------- | ------ | ---- | ------- | ---------- |
| MS Marco MiniLM l-12   | 1.00     | 1.00      | 0.99   | 1.00 | 0.01170 | 14s        |
| STS RoBERTa-base       | 0.98     | 0.99      | 0.96   | 0.98 | 0.51308 | 38s        |
| STS DistilRoBERTa-base | 1.00     | 1.00      | 0.99   | 1.00 | 0.50516 | 21s        |

### Amazon-Google - Test CompER Set - SBERT Model

| Model                                 | Accuracy | Precision | Recall | F1    | Loss     | Time/epoch |
| ------------------------------------- | -------- | --------- | ------ | ----- | -------- | ---------- |
| cross-encoder/ms-marco-MiniLM-L-12-v2 | 0.949    | 0.801     | 0.883  | 0.840 | 0.187761 | 49s        |
| cross-encoder/stsb-roberta-base       | 0.961    | 0.852     | 0.898  | 0.875 | 0.655815 | 136s       |
| cross-encoder/stsb-distilroberta-base | 0.964    | 0.871     | 0.898  | 0.885 | 0.656515 | 70s        |

### Abt-Buy - Test CompER Set - SBERT Model

| Model                                 | Accuracy | Precision | Recall | F1    | Loss     | Time/epoch |
| ------------------------------------- | -------- | --------- | ------ | ----- | -------- | ---------- |
| cross-encoder/ms-marco-MiniLM-L-12-v2 | 0.961    | 0.886     | 0.853  | 0.869 | 0.127730 | 18s        |
| cross-encoder/stsb-roberta-base       | 0.982    | 0.971     | 0.908  | 0.938 | 0.643272 | 54s        |
| cross-encoder/stsb-distilroberta-base | 0.975    | 0.950     | 0.881  | 0.914 | 0.646914 | 28s        |

### Amazon-Google - Test CompER Set (with 50 50 split) - SBERT Model

| Model                                 | Accuracy | Precision | Recall | F1    | Loss     | Time/epoch |
| ------------------------------------- | -------- | --------- | ------ | ----- | -------- | ---------- |
| cross-encoder/ms-marco-MiniLM-L-12-v2 | 0.949    | 0.770     | 0.954  | 0.852 | 0.323401 | 7s         |
| cross-encoder/stsb-roberta-base       | 0.973    | 0.857     | 0.991  | 0.919 | 0.651885 | 20s        |
| cross-encoder/stsb-distilroberta-base | 0.966    | 0.846     | 0.954  | 0.897 | 0.653681 | 10s        |

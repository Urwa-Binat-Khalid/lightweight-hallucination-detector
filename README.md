# Lightweight Hallucination Detector

A parameter-efficient hallucination detection system using a frozen RoBERTa encoder with a BiGRU classification head.

This project explores whether a frozen pretrained transformer can achieve hallucination detection performance comparable to fully fine-tuned models while updating less than 1% of total parameters.

---

# Research Problem

Large language model based hallucination detectors achieve strong performance but require updating hundreds of millions of parameters during fine-tuning.

This project investigates whether a lightweight architecture using frozen pretrained representations can maintain competitive accuracy with significantly fewer trainable parameters.

---

# Model Architecture

Frozen RoBERTa Encoder → BiGRU → Linear Classification Head

Only the BiGRU and classifier layers are trainable.

---

# Dataset

HaluEval Benchmark Dataset

---

# Results

| Model | F1 Score |
|---|---|
| BERT Fine-tuned | 97.53% |
| RoBERTa Fine-tuned | 97.53% |
| Frozen RoBERTa + Linear | 67.38% |
| Frozen RoBERTa + BiLSTM | 96.80% |
| Frozen RoBERTa + BiGRU (Proposed) | 97.07% |

The proposed architecture achieves near fine-tuned performance while updating less than 1% of total model parameters.

---

# Features

- Frozen pretrained transformer encoder
- Parameter-efficient training
- Lightweight trainable architecture
- Near fine-tuned performance
- PyTorch implementation

---

# Project Structure

```text
lightweight-hallucination-detector/
│
├── README.md
├── requirements.txt
├── train.py
├── inference.py
├── models.py
├── utils.py
├── config.py
├── notebooks/
├── results/
└── saved_models/
```

# Installation

```bash
pip install -r requirements.txt
```

# Run Training

```bash
python train.py
```

# Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Pandas
- NumPy

# Author

Urwa

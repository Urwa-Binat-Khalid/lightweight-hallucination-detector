import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

from config import CFG
from models import FrozenRobertaBiGRU
from utils import HallucinationDataset, set_seed, compute_metrics, count_parameters


def load_halueval_dataset():
    dataset = load_dataset("flowaicom/HaluEval", split="test")

    rows = []

    for example in dataset:
        passage = (example["passage"] or "").strip()[:300]
        question = (example["question"] or "").strip()
        answer = (example["answer"] or "").strip()

        text = f"Passage: {passage} Question: {question} Answer: {answer}"

        label = 1 if example["label"].strip().upper() == "FAIL" else 0
        rows.append((text, label))

    dataframe = pd.DataFrame(rows, columns=["text", "label"])
    dataframe = dataframe.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)

    return dataframe


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()

    total_loss = 0
    predictions = []
    labels_list = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(CFG.device)
        attention_mask = batch["attention_mask"].to(CFG.device)
        labels = batch["label"].to(CFG.device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        batch_predictions = torch.argmax(logits, dim=1)

        predictions.extend(batch_predictions.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

    metrics = compute_metrics(labels_list, predictions)

    return total_loss / len(dataloader), metrics


def evaluate(model, dataloader, criterion):
    model.eval()

    total_loss = 0
    predictions = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(CFG.device)
            attention_mask = batch["attention_mask"].to(CFG.device)
            labels = batch["label"].to(CFG.device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            batch_predictions = torch.argmax(logits, dim=1)

            predictions.extend(batch_predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    metrics = compute_metrics(labels_list, predictions)

    return total_loss / len(dataloader), metrics


def main():
    set_seed(CFG.seed)
    os.makedirs(CFG.save_dir, exist_ok=True)

    dataframe = load_halueval_dataset()

    train_df, temp_df = train_test_split(
        dataframe,
        test_size=0.30,
        random_state=CFG.seed,
        stratify=dataframe["label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=CFG.seed,
        stratify=temp_df["label"]
    )

    tokenizer = RobertaTokenizer.from_pretrained(CFG.model_name)

    train_dataset = HallucinationDataset(train_df, tokenizer, CFG.max_len)
    val_dataset = HallucinationDataset(val_df, tokenizer, CFG.max_len)
    test_dataset = HallucinationDataset(test_df, tokenizer, CFG.max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False
    )

    model = FrozenRobertaBiGRU(
        model_name=CFG.model_name,
        num_classes=CFG.num_classes,
        hidden_dim=CFG.rnn_hidden,
        num_layers=CFG.rnn_layers,
        dropout=CFG.rnn_dropout
    ).to(CFG.device)

    total_params, trainable_params = count_parameters(model)

    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    print("Trainable percentage:", round((trainable_params / total_params) * 100, 4))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.learning_rate
    )

    best_val_f1 = 0

    for epoch in range(CFG.epochs):
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            criterion
        )

        print(f"Epoch {epoch + 1}/{CFG.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val F1: {val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), CFG.best_model_path)

    model.load_state_dict(torch.load(CFG.best_model_path, map_location=CFG.device))

    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        criterion
    )

    print("Final Test Results")
    print(test_metrics)


if __name__ == "__main__":
    main()

import argparse

import os

import torch

import torch.nn as nn

import torch.optim as optim

from sklearn.metrics import roc_curve, auc, classification_report

from tqdm import tqdm

import pandas as pd

import numpy as np

import pickle as pkl

from transformers import BertModel, BertConfig, AdamW

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length", type=int, default=100, help="max sequence length.")
parser.add_argument("--data_dir", type=str, default='./inter_data/', help="data dir.")

parser.add_argument("--train_input_file", type=str, required=True, help="Input train file for finetuning")

parser.add_argument("--test_input_file", type=str, required=True, help="Input test file for finetuning")

parser.add_argument("--vocab_filename", type=str, required=True, help="Vocab filename")

parser.add_argument("--bert_config_file", type=str, required=True, help="BERT config file")

parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size for downside MLP.")

parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")

parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size")

parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")

parser.add_argument("--bizdate", type=str, required=True, help="Bizdate for the run")

args = parser.parse_args()

class FinetuneModel(nn.Module):

    def __init__(self, bert_config, hidden_size):
        super(FinetuneModel, self).__init__()

        self.bert = BertModel(bert_config)

        self.dnn1 = nn.Linear(bert_config.hidden_size, hidden_size)

        self.dnn2 = nn.Linear(hidden_size, hidden_size)

        self.logit = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, input_positions, input_io_flags, input_values, input_counts):
        transformer_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        inp = transformer_output[:, 0, :]  # Taking the [CLS] token representation

        dnn1_out = torch.relu(self.dnn1(inp))

        dnn2_out = torch.relu(self.dnn2(dnn1_out))

        logit = self.logit(dnn2_out + dnn1_out)

        return torch.sigmoid(logit).squeeze(-1)


def load_data(filepath, vocab, max_seq_length):
    with open(filepath, "rb") as f:
        data = pkl.load(f)

    sequences = []

    labels = []

    for seq in data:
        input_ids = vocab.convert_tokens_to_ids([token[0] for token in seq])

        input_ids = input_ids[:max_seq_length] + [0] * (max_seq_length - len(input_ids))

        sequences.append(input_ids)

        labels.append(seq[0][1])

    return torch.tensor(sequences), torch.tensor(labels, dtype=torch.float32)


def train(args, model, train_dataloader, optimizer, device):
    model.train()

    losses = []

    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()

        input_ids, input_mask, input_positions, input_io_flags, input_values, input_counts, labels = [b.to(device) for b
                                                                                                      in batch]

        outputs = model(input_ids, input_mask, input_positions, input_io_flags, input_values, input_counts)

        loss = nn.BCELoss()(outputs, labels)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    print(f"Training loss: {np.mean(losses)}")


def evaluate(args, model, eval_dataloader, device):
    model.eval()

    y_hat_list = []

    label_list = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            input_ids, input_mask, input_positions, input_io_flags, input_values, input_counts, labels = [b.to(device)
                                                                                                          for b in
                                                                                                          batch]

            outputs = model(input_ids, input_mask, input_positions, input_io_flags, input_values, input_counts)

            y_hat_list.extend(outputs.cpu().numpy())

            label_list.extend(labels.cpu().numpy())

    return np.array(y_hat_list), np.array(label_list)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data

    train_data, train_labels = load_data(args.train_input_file, args.vocab, args.max_seq_length)

    test_data, test_labels = load_data(args.test_input_file, args.vocab, args.max_seq_length)

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)

    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    # Load model

    bert_config = BertConfig.from_pretrained(args.bert_config_file)

    model = FinetuneModel(bert_config, args.hidden_size)

    model.to(device)

    # Optimizer

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Training

    train(args, model, train_dataloader, optimizer, device)

    # Evaluation

    y_hat, labels = evaluate(args, model, test_dataloader, device)

    # Aggregation and final evaluation

    fpr, tpr, thresholds = roc_curve(labels, y_hat)

    print(f"AUC: {auc(fpr, tpr)}")

    for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        print(f"Threshold = {threshold}")

        y_pred = (y_hat >= threshold).astype(int)

        print(classification_report(labels, y_pred, digits=4))


if __name__ == '__main__':


    # Load vocab

    vocab_file_name = os.path.join(args.data_dir, args.vocab_filename + "." + args.bizdate)

    with open(vocab_file_name, "rb") as f:
        args.vocab = pkl.load(f)

    main(args)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import argparse

from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split, KFold

import os
import datetime

device = 'cuda'

class Dataset(Dataset):
  def __init__(self, labels_df):
    self.labels_df = labels_df
    self.input_files = labels_df['file'].values
    self.labels = labels_df['label'].values
    self.xs = np.zeros(shape=(len(self.labels), 98346))
    for idx, file in enumerate(self.input_files):
      x = np.load(f'abide_fc_dataset/input/{file}')
      x = x[np.triu_indices(444, k=1)]
      self.xs[idx] = x

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    x = torch.from_numpy(self.xs[idx]).float()
    y = torch.tensor([self.labels[idx]]).float()
    y -= 1 # 1 -> 0, 2 -> 1
    return x, y

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    layers = [512, 256, 32]
    hidden_layers = []
    for i in range(len(layers) - 1):
      hidden_layers.append(nn.Linear(layers[i], layers[i + 1]))
      hidden_layers.append(nn.ReLU())

    self.net = nn.Sequential(
      nn.Linear(98346, layers[0]),
      nn.ReLU(),
      *hidden_layers,
      nn.Linear(32, 1)
    )

  def forward(self, x):
    x = self.net(x)
    #x = torch.sigmoid(x)
    return x

def train(model, train_loader, optimizer, criterion):
  model.train()
  total_loss = 0
  acc = 0
  for i, (x, y) in enumerate(train_loader):
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

    y_pred = torch.sigmoid(y_pred) > 0.5
    y_pred = y_pred.int()
    y_pred = y_pred == y
    y_pred = y_pred.int()
    acc += y_pred.sum().item()

  return total_loss / len(train_loader), acc / len(train_loader)

def evaluate(model, val_loader, criterion):
  model.eval()
  total_loss = 0
  correct_count = 0

  with torch.no_grad():
    for i, (x, y) in enumerate(val_loader):
      x = x.to(device)
      y = y.to(device)
      y_pred = model(x)
      loss = criterion(y_pred, y)
      total_loss += loss.item()

      preds = y_pred > 0
      correct_count += (preds == y).sum().item()

  accuracy = correct_count / len(val_loader.dataset)
  return total_loss / len(val_loader), accuracy

def get_predictions(model, test_loader):
  model.eval()
  predictions = []
  with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
      y_pred = model(x)
      y_pred = y_pred.squeeze(1)
      y_pred = y_pred > 0.5
      y_pred = y_pred.int()
      y_pred += 1
      predictions.extend(y_pred.tolist())
  return predictions

def train_model(train_loader, valid_loader, save_file, epochs, lr, decay):
  model = Model()
  print(model)
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
  criterion = nn.BCEWithLogitsLoss()

  best_loss = float('inf')
  for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
    print(f'Epoch: {epoch}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, Valid Acc: {valid_acc}')

    if valid_loss < best_loss:
      best_loss = valid_loss
      print('Saving new best model...')
      torch.save(model.state_dict(), save_file)

def train_fold(labels, date, fold, train_idx, valid_idx, batch_size, lr, decay):
  print(f'Fold: {fold}')
  labels_train = labels.iloc[train_idx]
  labels_valid = labels.iloc[valid_idx]

  train_loader = DataLoader(Dataset(labels_train), batch_size=batch_size, shuffle=False, num_workers=0)
  valid_loader = DataLoader(Dataset(labels_valid), batch_size=batch_size, shuffle=False, num_workers=0)

  train_model(train_loader, valid_loader, f'model-weights/nn-{date}-fold-{fold}.pth', epochs=100, lr=lr, decay=decay)

def main(lr, batch_size, decay):
  labels = pd.read_csv('abide_fc_dataset/labels_train.csv')
  date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

  # k = 5
  # kf = KFold(n_splits=k, shuffle=True, random_state=2023)
  # for fold, (train_idx, valid_idx) in enumerate(kf.split(labels)):
  #   train_fold(labels, date, fold, train_idx, valid_idx)

  dataset_train_all = Dataset(labels)
  train_loader_all = DataLoader(dataset_train_all, batch_size=batch_size, shuffle=False, num_workers=0)
  valid_loader_all = DataLoader(dataset_train_all, batch_size=batch_size, shuffle=False, num_workers=0)
  train_model(train_loader_all, valid_loader_all, f'model_weights/nn-{date}-all.pth', epochs=60, lr=lr, decay=decay)

if __name__ == '__main__':
  args = argparse.ArgumentParser()
  args.add_argument('--lr', type=float, default=1e-4)
  args.add_argument('--batch_size', type=int, default=128)
  args.add_argument('--decay', type=float, default=0.02)
  args = args.parse_args()
  main(**vars(args))
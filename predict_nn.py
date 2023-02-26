import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_nn import Dataset, Model

from sklearn.model_selection import train_test_split, KFold

import os

device = 'cuda'

labels_train = pd.read_csv('abide_fc_dataset/labels_train.csv')
labels_test = pd.read_csv('abide_fc_dataset/labels_test.csv')

def get_predictions(model, loader):
  model.eval()
  predictions = []
  with torch.no_grad():
    for x, y in loader:
      x = x.to(device)
      y = y.to(device)
      y_pred = model(x)
      y_pred = torch.sigmoid(y_pred)
      y_pred = y_pred.cpu().numpy().flatten()
      predictions.extend(y_pred)
  return predictions

predictions_valid = pd.DataFrame(columns=['file', 'label', 'prediction'])
predictions_train = pd.DataFrame(columns=['file', 'label', 'prediction'])
predictions_test = pd.DataFrame(columns=['file', 'label', 'prediction'])

# cross validation prediction
kf = KFold(n_splits=5, shuffle=True, random_state=2023)
for fold, (train_idx, valid_idx) in enumerate(kf.split(labels_train)):
  print(f'Fold: {fold}')
  labels_train_fold = labels_train.iloc[train_idx]
  labels_valid_fold = labels_train.iloc[valid_idx]

  train_loader = DataLoader(Dataset(labels_train_fold), batch_size=32, shuffle=False, num_workers=0)
  valid_loader = DataLoader(Dataset(labels_valid_fold), batch_size=32, shuffle=False, num_workers=0)

  model = Model()
  model.to(device)
  model.load_state_dict(torch.load(f'model_weights/nn-2023-02-25_19-24-41-fold-{fold}.pt'))

  predictions = get_predictions(model, valid_loader)
  labels_valid_fold['prediction'] = predictions
  predictions_valid = predictions_valid.append(labels_valid_fold)

# test prediction
test_loader = DataLoader(Dataset(labels_test), batch_size=32, shuffle=False, num_workers=0)
model = Model()
model.to(device)
model.load_state_dict(torch.load(f'model_weights/nn-2023-02-26_07-42-33-all.pth'))
predictions = get_predictions(model, test_loader)
predictions_test['file'] = labels_test['file'].values
predictions_test['label'] = labels_test['label'].values
predictions_test['prediction'] = predictions

# train prediction
train_loader = DataLoader(Dataset(labels_train), batch_size=32, shuffle=False, num_workers=0)
predictions = get_predictions(model, train_loader)
predictions_train['file'] = labels_train['file'].values
predictions_train['label'] = labels_train['label'].values
predictions_train['prediction'] = predictions

predictions_valid.to_csv('abide_fc_dataset/nn_predictions_valid.csv', index=False)
predictions_train.to_csv('abide_fc_dataset/nn_predictions_train.csv', index=False)
predictions_test.to_csv('abide_fc_dataset/nn_predictions_test.csv', index=False)
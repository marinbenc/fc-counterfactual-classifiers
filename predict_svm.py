import numpy as np
import pandas as pd
import pickle
import os
from joblib import dump, load
import datetime

from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.model_selection import train_test_split

def get_dataset(labels_df):
  ys = labels_df['label'].values
  xs = np.zeros(shape=(len(ys), 98346))

  input_files = labels_df['file'].values

  for i in range(len(input_files)):
    x = np.load(f'abide_fc_dataset/input/{input_files[i]}')
    x = x[np.triu_indices(444, k=1)]
    xs[i] = x

  return xs, ys

labels_train = pd.read_csv('abide_fc_dataset/labels_train.csv')
labels_test = pd.read_csv('abide_fc_dataset/labels_test.csv')

set_names = ['train', 'test']
sets = [labels_train, labels_test]

model = load(f'model_weights/svm-2023-02-24_10-51-39.joblib')

for i in range(len(sets)):
  xs, ys = get_dataset(sets[i])
  classes = model.predict(xs)
  results_df = pd.DataFrame(columns=['file', 'label', 'prediction'])
  results_df['file'] = sets[i]['file']
  results_df['label'] = sets[i]['label']
  results_df['prediction'] = classes
  results_df.to_csv(f'abide_fc_dataset/predictions_{set_names[i]}.csv', index=False)
  
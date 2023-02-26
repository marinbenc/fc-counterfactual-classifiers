import numpy as np
import pandas as pd
import pickle
import os
from joblib import dump, load
import datetime

from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.model_selection import train_test_split

labels_train = pd.read_csv('abide_fc_dataset/labels_train.csv')

ys = labels_train['label'].values
xs = np.zeros(shape=(len(ys), 98346))

input_files = labels_train['file'].values

for i in range(len(input_files)):
  x = np.load(f'abide_fc_dataset/input/{input_files[i]}')
  x = x[np.triu_indices(444, k=1)]
  xs[i] = x

model = SVC(kernel='rbf', C=100, gamma='auto', random_state=2023, probability=True)
model.fit(xs, ys)

dirname = os.path.dirname(__file__)
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
dump(model, f'model_weights/svm-{date}.joblib')

scores = cross_val_score(model, xs, ys, cv=5)
print(scores)

import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

labels_train = pd.DataFrame(columns=['file', 'label'])
labels_test = pd.DataFrame(columns=['file', 'label'])

scales = [444, 122]
scales = [f'scale{scale}' for scale in scales]

for scale in scales:
  if scale == 'scale444':
    continue
  folder = Path(f'abide_fc_dataset_{scale}')
  phenotypic = pd.read_csv(folder/'Phenotypic_V1_0b_preprocessed1.csv')
  input_files = os.listdir(folder/'input')

  # train, valid, test
  train, test = train_test_split(input_files, test_size=0.2, random_state=2023)

  for i in range(len(train)):
    file = train[i]
    file_id = file.split('.')[0]
    label = phenotypic.loc[phenotypic['FILE_ID'] == file_id, 'DX_GROUP'].values[0]
    labels_train = labels_train.append({'file': file, 'label': label}, ignore_index=True)

  for i in range(len(test)):
    file = test[i]
    file_id = file.split('.')[0]
    label = phenotypic.loc[phenotypic['FILE_ID'] == file_id, 'DX_GROUP'].values[0]
    labels_test = labels_test.append({'file': file, 'label': label}, ignore_index=True)

  labels_train.to_csv(folder/'labels_train.csv', index=False)
  labels_test.to_csv(folder/'labels_test.csv', index=False)
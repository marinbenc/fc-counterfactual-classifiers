import os
import pandas as pd
from sklearn.model_selection import train_test_split

labels_train = pd.DataFrame(columns=['file', 'label'])
labels_test = pd.DataFrame(columns=['file', 'label'])

phenotypic = pd.read_csv('abide_fc_dataset/Phenotypic_V1_0b_preprocessed1.csv')
input_files = os.listdir('abide_fc_dataset/input')

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

labels_train.to_csv('abide_fc_dataset/labels_train.csv', index=False)
labels_test.to_csv('abide_fc_dataset/labels_test.csv', index=False)
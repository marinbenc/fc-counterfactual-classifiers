import numpy as np
import pandas as pd
import pickle
import os
from joblib import dump, load

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# calculate nn metrics

predictions_test = pd.read_csv('abide_fc_dataset/nn_predictions_test.csv')
predictions_train = pd.read_csv('abide_fc_dataset/nn_predictions_train.csv')
predictions_valid = pd.read_csv('abide_fc_dataset/nn_predictions_valid.csv')

def p_to_prediction(p):
  if p < 0.5:
    return 0
  else:
    return 1

predictions_test['predicted_class'] = predictions_test['prediction'].apply(p_to_prediction)

print(f'NN test (n = {len(predictions_test)})')
test_cm = confusion_matrix(predictions_test['label'] - 1, predictions_test['predicted_class'], labels=[0, 1])
test_acc = accuracy_score(predictions_test['label'] - 1, predictions_test['predicted_class'])

def sensitivity(cm):
  return cm[1][1] / (cm[1][1] + cm[1][0])

def specificity(cm):
  return cm[0][0] / (cm[0][0] + cm[0][1])

test_sens = sensitivity(test_cm)
test_spec = specificity(test_cm)
test_f1 = f1_score(predictions_test['label'] - 1, predictions_test['predicted_class'])

print(f'Accuracy: {test_acc}')
print(f'Sensitivity: {test_sens}')
print(f'Specificity: {test_spec}')
print(f'F1 score: {test_f1}')

print('----------------------------------------')
print(f'NN 5-fold cross validation (n = {len(predictions_valid)})')

predictions_valid['predicted_class'] = predictions_valid['prediction'].apply(p_to_prediction)
valid_cm = confusion_matrix(predictions_valid['label'] - 1, predictions_valid['predicted_class'], labels=[0, 1])
valid_acc = accuracy_score(predictions_valid['label'] - 1, predictions_valid['predicted_class'])
valid_sens = sensitivity(valid_cm)
valid_spec = specificity(valid_cm)
valid_f1 = f1_score(predictions_valid['label'] - 1, predictions_valid['predicted_class'])

print(f'Accuracy: {valid_acc}')
print(f'Sensitivity: {valid_sens}')
print(f'Specificity: {valid_spec}')
print(f'F1 score: {valid_f1}')


predictions_test = pd.read_csv('abide_fc_dataset/svm_predictions_test.csv')

print('----------------------------------------')
print(f'SVM test (n = {len(predictions_test)})')

test_cm = confusion_matrix(predictions_test['label'], predictions_test['prediction'], labels=[1, 2])
test_acc = accuracy_score(predictions_test['label'], predictions_test['prediction'])
test_sens = sensitivity(test_cm)
test_spec = specificity(test_cm)
test_f1 = f1_score(predictions_test['label'], predictions_test['prediction'])

print(f'Accuracy: {test_acc}')
print(f'Sensitivity: {test_sens}')
print(f'Specificity: {test_spec}')
print(f'F1 score: {test_f1}')
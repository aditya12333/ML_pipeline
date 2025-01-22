import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Load the model
clf = pickle.load(open('model.pkl', 'rb'))

# Load the test data
test_data = pd.read_csv('data/features/test_bow.csv')

X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values

# Map string labels to integers
label_mapping = {'happiness': 1, 'sadness': 0}
y_test = np.array([label_mapping[label] for label in y_test])
y_pred = np.array([label_mapping[label] for label in clf.predict(X_test)])
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Save metrics to a JSON file
metrics_dict = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'auc': auc
}

with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)

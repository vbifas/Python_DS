import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

perceptron_test = pd.read_csv( 'test.csv', names=['1', '2', '3'])
perceptron_train = pd.read_csv( 'train.csv', names=['1', '2', '3'])

X_train = perceptron_train[['2', '3']]
y_train = perceptron_train['1']
X_test = perceptron_test[['2', '3']]
y_test = perceptron_test['1']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = Perceptron(random_state=241,max_iter=5, tol=None)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
score = accuracy_score(y_test, y_pred)
print(score)

import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


X, y = load_boston(return_X_y=True)
X = sklearn.preprocessing.scale(X)
kf = KFold(n_splits=5, random_state=42, shuffle=True)
neigh = KNeighborsRegressor(n_neighbors=5, weights='distance')
listM = []
i = np.linspace(1, 10, num=200)
for p in i:
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, metric='minkowski')
    neigh.fit(X, y)
    mass = cross_val_score(neigh, X=X, y=y, scoring='neg_mean_squared_error', cv=kf)
    m = np.mean(mass)
    listM.append((m, p))
m = max(listM)
print(m)

import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

wine_priznak = pd.read_csv('wine.csv', names=['Type', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity of ash',
                                              'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                                              'Proanthocyanins', 'Color intensity', 'Hue',
                                              'OD280/OD315 of diluted wines',
                                              'Proline'])
X = wine_priznak.drop(wine_priznak.columns[0], axis='columns')
y = wine_priznak['Type']
X = sklearn.preprocessing.scale(X)
kf = KFold(n_splits=5, random_state=42, shuffle=True)
listM = []
for k in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y)
    mass = cross_val_score(neigh, X=X, y=y, cv=kf)
    m = np.mean(mass)
    listM.append((m, k))
m = max(listM)
print(m)

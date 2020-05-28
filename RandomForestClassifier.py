import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

data = pd.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else(
    -1 if x == 'F' else 0))
X = data.iloc[:, 0:7]
y = data['Rings']
a = []

kf = KFold(n_splits=5, random_state=42, shuffle=True)
kf.get_n_splits(X)
for i in range(1, 51):
    reg = RandomForestRegressor(n_estimators=i, random_state=42, n_jobs=-1)
    reg.fit(X, y)
    score = cross_val_score(reg, X=X, y=y, scoring='r2', cv=kf)
    res = round(score.mean(), 3)
    print(i, res)
    if res > 0.52:
        a.append(i)
print(min(a))

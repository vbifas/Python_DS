import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

data_close = pd.read_csv('close_prices.csv')
data_index = pd.read_csv('djia_index.csv')

X = data_close.iloc[:, 1:31]
y = data_index['^DJI']

pca = PCA(n_components=10)

comp = pca.explained_variance_ratio_
a = 0
i = 0
while a <= 0.9:
    a += comp[i]
    i += 1
pca = PCA(n_components=i)
first_comp = pd.DataFrame(pca.transform(X)[:, 0])
corr = np.corrcoef(y, first_comp.T)
np.argmax(pca.components_[0])

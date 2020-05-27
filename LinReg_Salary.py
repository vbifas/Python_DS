import numpy as np
from scipy.sparse import coo_matrix, hstack
from sklearn.linear_model import Ridge
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer


data_train = pd.read_csv('salary-train.csv')
y_train = data_train['SalaryNormalized']
data_train['FullDescription'] = data_train['FullDescription'].apply(lambda x: x.lower())
data_train['LocationNormalized'] = data_train['LocationNormalized'].apply(lambda x: x.lower())
data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
vectorizer = TfidfVectorizer(min_df=5)
X_desc = vectorizer.fit_transform(data_train['FullDescription'])
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X = hstack([X_desc, X_train_categ])
regreg = Ridge(random_state=241, alpha=1)
regreg.fit(X, y_train)

data_test = pd.read_csv('salary-test-mini.csv')

data_test['FullDescription'] = data_test['FullDescription'].apply(lambda x: x.lower())
data_test['LocationNormalized'] = data_test['LocationNormalized'].apply(lambda x: x.lower())
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)
X_test1 = vectorizer.transform(data_test['FullDescription'])
X_test2 = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test1, X_test2])
res = regreg.predict(X_test)
print(res)


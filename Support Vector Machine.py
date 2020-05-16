import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv('svm-data.csv', names=['1', '2', '3'])
data.head()
X = data[['2', '3']]
y = data['1']

clf = SVC(C=1000, kernel='linear', random_state=241)
clf.fit(X, y)
print(clf.support_ + 1)

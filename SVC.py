import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target
feature_mapping = vectorizer.get_feature_names()

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', n_jobs=-1, cv=cv)
gs.fit(X, y)

C_best = gs.best_params_['C']

clf = SVC(C=1, kernel='linear', random_state=241)
clf.fit(X, y)

weights = np.absolute(clf.coef_.toarray())
max_weights = sorted(zip(weights[0], feature_mapping))[-10:]
max_weights.sort(key=lambda x: x[1])
print(max_weights)

file = open('some.txt', 'w')
for i in range(10):
    file.write(max_weights[i][-1] + ' ')
file.close()

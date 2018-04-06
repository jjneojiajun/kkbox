import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

train = pd.read_csv("sorted_train.csv")
train = train.drop('Unnamed: 0', axis=1)

test = pd.read_csv("sorted_test.csv")
test = test.drop('Unnamed: 0', axis=1)

cols = [c for c in train.columns if c not in ['is_churn', 'msno']]

train = train.reset_index()
test = train.reset_index()

y = train['is_churn']
X_train, X_test, y_train, y_test = train_test_split(train[cols], y, test_size=0.3, random_state = 101)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=250, criterion = 'entropy', random_state= 101, verbose=0, n_jobs=1, oob_score=False, max_features='auto', min_samples_leaf=1, min_samples_split=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)


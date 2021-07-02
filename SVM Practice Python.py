import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

df = pd.read_csv('diabetes.csv')


df.dropna(inplace=True)

from sklearn.datasets import make_regression
y = df.pop('Outcome')
X = df
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40)

model = SVC()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear']}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_estimator_)


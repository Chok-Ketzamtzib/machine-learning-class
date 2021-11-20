# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 23:01:45 2021

@author: wakef
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import RocCurveDisplay
import seaborn as sns

haberman = pd.read_csv('haberman_data.csv.txt')
haberman.columns = ['Age', 'Year of Operation', 'Positive Axillary Nodes', 'Survival Status']


X = haberman[['Age', 'Year of Operation', 'Positive Axillary Nodes']]
y = haberman['Survival Status']

correlation_matrix = X.astype(float).corr()
sns.heatmap(correlation_matrix, annot = True)
plt.show()

print(haberman)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, train_size=.8, random_state=0)

lr = LogisticRegression()

lr.fit(X_train, y_train)

accuracy = lr.score(X_test, y_test)

print('Mean Accuracy: ', accuracy)

y_pred = lr.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['negative class', 'positive class']))

fpr, tpr, thresholds = roc_curve(y_test, lr.decision_function(X_test), pos_label=2)

print('fpr: ', fpr)
print('tpr: ', tpr)
print('thresholds: ', thresholds)

close_zero = np.argmin(np.abs(thresholds))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=7, label="Best", c='k', mew=2)
plt.plot([0,1], 'b--', label='No Skill')
plt.plot(fpr, tpr, color='darkorange', marker='o', markersize=4, label="Logistic")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
plt.legend(loc=4)
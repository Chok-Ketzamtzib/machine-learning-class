# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:38:58 2021

@author: William J. Wakefield and Brandon Peterson
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

machine = pd.read_csv('machine_data.csv.data')
machine.columns = ['Vendor Name', 'Model Name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']

X = machine[['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX']]
y = machine['PRP']
# split data into train+validation set and test set
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, random_state=1)
# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, random_state=3)
print("Size of training set: {}   size of validation set: {}   size of test set:"
      " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

ridge = Ridge().fit(X_train, y_train)
print("Score without interactions Ridge: {:.3f}".format(ridge.score(X_test, y_test)))

lasso = Lasso().fit(X_train, y_train)
print("Score without interactions Lasso: {:.3f}".format(lasso.score(X_test, y_test)))

'''
Part C
'''
param_grid = {'alpha': [0, 0.1, 1.0, 10.0, 20.0, 50.0, 100.0]}

grid_search_ridge = GridSearchCV(ridge, param_grid, cv=5)
grid_search_lasso = GridSearchCV(lasso, param_grid, cv=5)

#Do the fitting per regression method after prerforming grid search
grid_search_ridge.fit(X_train, y_train)
grid_search_lasso.fit(X_train, y_train)



print("Best Parameter for Ridge: \n", grid_search_ridge.best_params_)
print("Best Parameter for Lasso: \n", grid_search_lasso.best_params_)
best_param_ridge = grid_search_ridge.best_params_['alpha']
best_param_lasso = grid_search_lasso.best_params_['alpha']

'''
Part D
'''

lr = LinearRegression()
lr.fit(X_train, y_train)

pred_train_lr= lr.predict(X_train)
linear_RMSE = np.sqrt(mean_squared_error(y_train,pred_train_lr))
linear_r2 = r2_score(y_train, pred_train_lr)
print("Linear RMSE: ", linear_RMSE)
print("Linear R2 SCore: ", linear_r2)

rr = Ridge(alpha=best_param_ridge)
rr.fit(X_train, y_train) 
pred_train_rr= rr.predict(X_train)
ridge_RMSE = np.sqrt(mean_squared_error(y_train,pred_train_rr))
ridge_r2 = r2_score(y_train, pred_train_rr)
print("Ridge RMSE: ", ridge_RMSE)
print("Ridge R2 Score: ", ridge_r2)

model_lasso = Lasso(alpha=best_param_lasso)
model_lasso.fit(X_train, y_train) 
pred_train_lasso= model_lasso.predict(X_train)
lasso_RMSE = np.sqrt(mean_squared_error(y_train,pred_train_lasso))
lasso_r2 = r2_score(y_train, pred_train_lasso)
print("Lasso RMSE: ", lasso_RMSE)
print("Lasso R2 Score: ", lasso_r2)

'''
Part E
'''
plt.plot(ridge.coef_,linestyle='none',marker='s',markersize=6,color='red',label=r'Ridge; $\alpha = 0.01$', zorder = 7) 
plt.plot(rr.coef_,linestyle='none',marker='v',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') 
plt.plot(lasso.coef_,linestyle='none',marker='^',markersize=6,color='pink',label=r'Lasso; $\alpha = 0.01$')
plt.plot(model_lasso.coef_,linestyle='none',marker='d',markersize=6,color='purple',label=r'Lasso; $\alpha = 50$')
plt.plot(lr.coef_,linestyle='none',marker='o',markersize=6,color='green',label='Linear Regression')
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=10,loc=2)
plt.show()
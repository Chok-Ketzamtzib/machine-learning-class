#!/usr/bin/env python
# coding: utf-8

# In[56]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
#import mglearn
import math
from numpy import genfromtxt


# In[2]:


class CSV_Data:
   def __init__(self, data, target, feature_names, target_name, file_name):
        self.data = data
        self.target = target
        self.feature_names = feature_names
        self.target_name = target_name
        self.file_name = file_name
        

def loadData(file_path):
    data = []
    target = []
    feature_names = []
    target_name = ''
    file_name = file_path.split('\\')[-1]
    with open(file_path) as data_file:
        all_lines = [s[:-1] for s in data_file.readlines()]
        feature_names = all_lines[0].split(',')[1:]
        target_name = all_lines[0].split(',')[0]
        for line in all_lines[1:]:
            values = line.split(',')
            target.append(values[0])
            data.append([])
            for val in values:
                data[-1].append(val)
    return CSV_Data(np.array(data, ndmin=2), np.array(target), feature_names, target_name, file_name)

wine_data = loadData('Wine-A2\wine.csv')
print(wine_data.data)
print("Shape of wine data:",wine_data.data.shape)
print(wine_data.target)
print("Shape of wine target:", wine_data.target.shape)
print("wine feature names:", wine_data.feature_names)
print(wine_data.target_name)
print(wine_data.file_name)


# In[7]:


print("Shape of wine data:", wine_data.data.shape)
print("Shape of wine target:", wine_data.target.shape)
print("wine feature names:", wine_data.feature_names)
#print("Shape of wine target names:", wine_data.target_names)
#print("Shape of wine:", wine_data.frame)


# In[10]:


print("Number of Features:", len(wine_data.feature_names))


# In[7]:


#4
from sklearn.datasets import load_wine
wine_data = load_wine()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    wine_data['data'], wine_data['target'], random_state=80)


# In[8]:


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# In[9]:


print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# In[11]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# In[12]:


knn.fit(X_train, y_train)


# In[19]:


X_new = np.array([[4, 3, 1, 0.2,5,0.4,8,6,5,5,11,15,0.5]])
print("X_new.shape:", X_new.shape)


# In[20]:


prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:",
       wine_data['target_names'][prediction])


# In[21]:


y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)


# In[22]:


print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# In[26]:


# Neighbor of 1
X_train, X_test, y_train, y_test = train_test_split(
    wine_data['data'], wine_data['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# In[28]:


# Neighbor of 3
X_train, X_test, y_train, y_test = train_test_split(
    wine_data['data'], wine_data['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# In[41]:


# Neighbor of sqrt(n) + 3
X_train, X_test, y_train, y_test = train_test_split(
    wine_data['data'], wine_data['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=math.ceil(math.sqrt(13)+3))
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(
    wine_data.data, wine_data.target, stratify=wine_data.target, random_state=80)

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to sqrt(n)+3
neighbors_settings = range(1, math.ceil(math.sqrt(len(X_train))+3))

for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# In[68]:


#5
class sklearn.model_selection.StratifiedKFold(n_splits=5,*,shuffle=False, random_state=None)


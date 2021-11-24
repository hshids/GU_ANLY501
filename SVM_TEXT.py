#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Import Packages
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import re


# In[5]:


df= pd.read_csv("~/desktop/GGT/module6/CV_News.csv")
df.head(5)


# In[9]:


x = df.drop(columns=['LABEL'])
# Target variable
y = df.LABEL    
# split train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) 


# In[10]:


NB = MultinomialNB(alpha = 1)
NB.fit(X_train, y_train)


# In[11]:


y_hat = NB.predict(X_test)


# In[12]:


confusion_matrix(y_test, y_hat)


# In[13]:


accuracy_score(y_test, y_hat)


# In[14]:


### linear
param_grid = {'C': [0.1, 1, 2, 5],
              'kernel': ['linear']}


# In[19]:


tune = GridSearchCV(SVC(), param_grid, refit = True, cv = 2, verbose = 2, n_jobs = 2 )


# In[20]:


tune.fit(train, train_label)


# In[21]:


tune.best_params_


# In[22]:


y_hat = tune.predict(X_test)


# In[23]:


conf_matrix = confusion_matrix(y_test, y_hat)


# In[24]:


accuracy_score(y_test, y_hat)


# In[25]:


fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[27]:


### sigmoid
param_grid = {'C': [0.1, 1, 2, 5],
              'kernel': ['sigmoid']}
tune = GridSearchCV(SVC(), param_grid, refit = True, cv = 2, verbose = 2, n_jobs = 2 )
tune.fit(X_train, y_train)
tune.best_params_
y_hat = tune.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_hat)
accuracy_score(y_test, y_hat)


# In[28]:


fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[32]:


### precomputed
param_grid = {'C': [0.1, 1, 2, 5],
              'kernel': ['rbf']}
tune = GridSearchCV(SVC(), param_grid, refit = True, cv = 2, verbose = 2, n_jobs = 2 )
tune.fit(X_train, y_train)
tune.best_params_
y_hat = tune.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_hat)
accuracy_score(y_test, y_hat)


# In[33]:


fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[38]:





# In[40]:


## feature importance
from sklearn.svm import LinearSVC
svm = LinearSVC(C=0.1)
svm.fit(X_train,y_train)
f_importances(svm.coef_[0], X_train.columns)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import seaborn as sns


# In[2]:


df=pd.read_csv("/Users/hanjingshi/Desktop/GGT/module6/CV_News.csv")
df.head()


# In[3]:


x = df.drop(columns=['LABEL'])
# Target variable
y = df.LABEL    
# split train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) 


# In[7]:


### DRAW WORDCLOUD
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");
wordcloud = WordCloud(width = 3000, 
                      height = 2000, 
                      random_state=1, 
                      background_color='black',  
                      collocations=False, 
                      stopwords = STOPWORDS).generate(' '.join(x.columns))
plot_cloud(wordcloud)


# In[11]:


## Naive Bayes ##
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test,y_pred ))


# In[12]:


conf_matrix = confusion_matrix(y_test,y_pred)


# In[13]:


# Print the confusion matrix using Matplotlib
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.colorbar()
plt.show()
plt.savefig('confM1.png')


# In[20]:


# feature importance
feature_importances=pd.Series(clf.feature_importances_,index=x.columns)
feature_importances.sort_values(inplace=True,ascending=False)


# In[21]:


# plot the feature importance
plt.bar(range(20),feature_importances[:20]*100)
plt.xlabel('Feature')
plt.ylabel('Importance(%)')
plt.xticks(range(20),feature_importances.index[:20],rotation=60)
plt.title('Feature Importance')
plt.show()
plt.savefig('feature_imp_py.png')


# In[19]:


y_pred1 = clf.predict_proba(X_test)
plt.hist(y_pred1, bins = 100)
plt.title('Histogram of predicted probabilities')
plt.xlim(0,1)
plt.xlabel('Predicted probabilities')
plt.ylabel('Frequency')


# In[ ]:





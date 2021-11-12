#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics 
from sklearn.tree import export_graphviz
from IPython.display import Image  
import pydotplus
from six import StringIO
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier


# In[21]:


df=pd.read_csv("/Users/hanjingshi/Desktop/GGT/501data/module5/CV_News.csv")
df.head()


# In[22]:


x = df.drop(columns=['LABEL'])
# Target variable
y = df.LABEL    
# split train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) 


# In[23]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[24]:


def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");


# In[25]:


wordcloud = WordCloud(width = 3000, 
                      height = 2000, 
                      random_state=1, 
                      background_color='black',  
                      collocations=False, 
                      stopwords = STOPWORDS).generate(' '.join(x.columns))
plot_cloud(wordcloud)


# In[26]:


#####################
## Decision Tree 1 ##
#####################

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", 
                             max_depth=20,
                             splitter='best',  ## or "random" or "best"
                             min_samples_split=2, 
                             min_samples_leaf=1)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[27]:


# plot the decision tree
os.environ["PATH"] += os.pathsep + '/Users/hanjingshi/Desktop/GGT/501data/module5'
dot_data = StringIO()
export_graphviz(clf, out_file= dot_data, 
                         filled=True, rounded=True,
                special_characters=True, 
                feature_names = x.columns,
                class_names= clf.classes_,
                label='all') 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('DT1.png')
Image(graph.create_png())
graph.savefig


# In[28]:


# model evaluation: confusion metric
conf_matrix = confusion_matrix(y_test,y_pred)

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


# In[29]:


##DT2
clf = DecisionTreeClassifier(criterion="gini", 
                             max_depth=10,
                             splitter='best',  ## or "random" or "best"
                             min_samples_split=2, 
                             min_samples_leaf=1)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# plot the decision tree
os.environ["PATH"] += os.pathsep + '/Users/hanjingshi/Desktop/GGT/501data/module5'
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, 
                feature_names = x.columns,
                class_names= clf.classes_,
                label='all')
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('DT2.png')
Image(graph.create_png())


# In[30]:


# model evaluation: confusion metric
conf_matrix = confusion_matrix(y_test,y_pred)

# Print the confusion matrix using Matplotlib
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
plt.savefig('confM2.png')


# In[31]:


## DT3
clf = DecisionTreeClassifier(criterion="entropy", 
                             max_depth=5,
                             splitter='best',  ## or "random" or "best"
                             min_samples_split=2, 
                             min_samples_leaf=1)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[32]:


os.environ["PATH"] += os.pathsep + '/Users/hanjingshi/Desktop/GGT/501data/module5'
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, 
                feature_names = x.columns,
                class_names= clf.classes_,
                label='all')
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('DT3.png')
Image(graph.create_png())

# model evaluation: confusion metric
conf_matrix = confusion_matrix(y_test,y_pred)

# Print the confusion matrix using Matplotlib
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
plt.savefig('confM3.png')


# In[33]:


# feature importance
feature_importances=pd.Series(clf.feature_importances_,index=x.columns)
feature_importances.sort_values(inplace=True,ascending=False)

# plot the feature importance
plt.bar(range(20),feature_importances[:20]*100)
plt.xlabel('Feature')
plt.ylabel('Importance(%)')
plt.xticks(range(20),feature_importances.index[:20],rotation=60)
plt.title('Feature Importance')
plt.show()
plt.savefig('feature_imp_py.png')


# In[34]:


clf = RandomForestClassifier(max_depth=10, random_state=1,n_estimators=100)
clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# model evaluation: confusion metric
conf_matrix = confusion_matrix(y_test,y_pred)


# In[35]:


# Print the confusion matrix using Matplotlib
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
plt.savefig('confMRF.png')


# In[ ]:





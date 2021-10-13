#!/usr/bin/env python
# coding: utf-8

# In[176]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.cluster.hierarchy as hc
import re
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


# In[177]:


data = pd.read_csv('~/Desktop/CLEAN/CV_News.csv')
data.head(n = 5)


# In[179]:


def tokenize(string):
    tokens = re.sub(r"[^A-Za-z]", " ", string).lower().split()
    return tokens


# In[180]:


vectorizer = TfidfVectorizer(input='content',   ## 'content'
                        stop_words='english',  ## and, of, the, is, ...
                        tokenizer=tokenize
                        )


# In[181]:


X = vectorizer.fit_transform(data['LABEL'])
My_DF=pd.DataFrame(X.toarray())

MyColumnNames=vectorizer.get_feature_names()
My_DF.columns = MyColumnNames


# In[182]:


My_DF.columns[0:5]


# In[183]:


labels = data['LABEL']
df = data.drop("LABEL", axis=1)


# In[ ]:





# In[184]:


##         Look at best values for k
SS_dist = []

values_for_k=range(2,10)
#print(values_for_k)

for k_val in values_for_k:
    #print(k_val)
    k_means = KMeans(n_clusters=k_val)
    k_means = k_means.fit(df)
    SS_dist.append(k_means.inertia_)
    
#print(SS_dist)
#print(values_for_k)

plt.plot(values_for_k, SS_dist, 'bx-')
plt.xlabel('value')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method for optimal k Choice')
plt.show()


# In[185]:


####
# Look at Silhouette
##########################
Sih=[]
Cal=[]
k_range=range(2,10)

for k in k_range:
    k_means_n = KMeans(n_clusters=k)
    model = k_means_n.fit(df)
    Pred = k_means_n.predict(df)
    labels_n = k_means_n.labels_
    R1=metrics.silhouette_score(df, labels_n, metric = 'euclidean')
    R2=metrics.calinski_harabasz_score(df, labels_n)
    Sih.append(R1)
    Cal.append(R2)

print(Sih) ## higher is better
print(Cal) ## higher is better

fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.plot(k_range,Sih)
ax1.set_title("Silhouette")
ax1.set_xlabel("")
ax2.plot(k_range,Cal)
ax2.set_title("Calinski_Harabasz_Score")
ax2.set_xlabel("k values")


# In[186]:


df_normalized=(df - df.mean()) / df.std()
#print(df_normalized)
NumCols=df_normalized.shape[1]


# In[187]:


## PCA
# Instantiated my own copy of PCA
My_pca = PCA(n_components=2)  ## two prin columns

# Transpose DTM
df_normalized = np.transpose(df_normalized)
My_pca.fit(df_normalized)
KnownLabels = list(data.iloc[:,0])
# Reformat and view results
Comps = pd.DataFrame(My_pca.components_.T,
                     columns = ['PC%s' % _ for _ in range(2)],
                     index = df_normalized.columns
                     )
print(Comps)
#RowNames = list(Comps.index)
#print(RowNames)


# In[ ]:





# In[188]:


########################
## Look at 2D PCA clusters
############################################

plt.figure(figsize=(35,20))
plt.scatter(Comps.iloc[:,0], Comps.iloc[:,1], s=100, color="pink")

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("Scatter Plot Clusters PC 1 and 2",fontsize=20)

plt.show()


# In[ ]:





# In[189]:


##         DBSCAN
Comps = Comps.iloc[:,0:2]
MyDBSCAN = DBSCAN(eps=100, min_samples=2)
## eps:
    ## The maximum distance between two samples for 
    ##one to be considered as in the neighborhood of the other.
MyDBSCAN.fit_predict(Comps)
print(MyDBSCAN.labels_)


# In[190]:


Comps['label'] = MyDBSCAN.labels_
sns.lmplot(data=Comps, x='PC0', y='PC1', hue='label',
           fit_reg=False, legend=True, legend_out=True)
ax = plt.gca()
ax.set_title("DBSCAN Results")
plt.show()


# In[ ]:





# In[197]:


# Remove previously assigned label
Comps = Comps.iloc[:,0:2]
# Comps
for k in range(2,10):
    kmeans = sklearn.cluster.KMeans(n_clusters=k) 
    kmeans.fit(Comps)
# Get cluster assignment labels
    labels = kmeans.labels_
    #print(labels)
    prediction_kmeans = kmeans.predict(Comps)
    #print(prediction_kmeans2)
# Format results as a DataFrame
    Myresults = pd.DataFrame([Comps.index,labels]).T
    #print(Myresults2)

# plot cluster results for k=4
    Comps['label'] = labels1
    #print(labels2)
    sns.lmplot(data=Comps, x='PC0', y='PC1',hue='label'
            ,fit_reg=False, legend=True, legend_out=True)
    ax = plt.gca()
    ax.set_title("K-means Results When k=%d"%(k))
    plt.show()


# In[193]:


## Hierarchical
# Remove previously assigned label
Comps = Comps.iloc[:,0:2]
# Comps

MyHC = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
FIT = MyHC.fit(Comps)
HC_labels = MyHC.labels_
# print(HC_labels)

plt.figure(figsize =(12, 12))
plt.title('Hierarchical Clustering')
dendro = hc.dendrogram((hc.linkage(Comps, method ='ward')))


# In[ ]:





# In[ ]:





# In[ ]:





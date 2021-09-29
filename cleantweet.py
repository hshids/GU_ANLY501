#!/usr/bin/env python
# coding: utf-8

# In[16]:


from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import os
import re

import warnings
warnings.filterwarnings('ignore')


# In[17]:


tweet_DF=pd.read_csv("~/Desktop/CLEAN/tweetdata.csv", error_bad_lines=False)
tweet_DF.head()
tweet_DF.info()


# In[18]:


to_drop = ["favoriteCount", "id", "statusSource", "longitude", "latitude", "replyToSID","replyToUID"]
tweet_DF = tweet_DF.drop(to_drop, axis = 1)
tweet_DF = tweet_DF.drop("replyToSN", axis = 1)


# In[19]:


#pip install emoji


# In[5]:


"""
Remove emojis
"""
tweet_DF = tweet_DF.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
tweet_DF = tweet_DF.drop_duplicates()


# In[6]:


def remove(text):
    remove_chars = '[0-9!#$%&\'()*+,-./:;<=>?@,?[]<>''![\\]^_`{|}~]+'
    return re.sub(remove_chars, '', text)
res = []


# In[7]:


My_Content_List=["text"]
My_Labels_List=["favoriteCount"]


# In[8]:


MyCV_content=CountVectorizer(input='content',
                        stop_words='english',
                        #max_features=100
                        )


# In[9]:


My_DTM2=MyCV_content.fit_transform(My_Content_List)


# In[10]:


ColNames=MyCV_content.get_feature_names()
print("The vocab is: ", ColNames, "\n\n")


# In[11]:


My_DF_content=pd.DataFrame(My_DTM2.toarray(),columns=ColNames)


# In[12]:


print(My_DF_content)


# In[13]:


print(My_Labels_List)


# In[14]:


My_DF_content.to_csv('tweetclean.csv', index=False)


# In[15]:


#tweet_DF.to_csv('tweetclean.csv', index=False)


# In[ ]:

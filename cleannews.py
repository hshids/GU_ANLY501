#!/usr/bin/env python
# coding: utf-8

# In[13]:


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


# In[14]:


news_df=pd.read_csv("~/Desktop/GGT/newsapi.csv", error_bad_lines=False)
news_df.head()
news_df.info()


# In[15]:


to_drop = ["Author1","Description","V2","V3","V4","V5","V6","V7","V8","V9","V1O"]
news_df = news_df.drop(to_drop, axis = 1)


# In[16]:


news_df.head()


# In[22]:


news_df.to_csv("cleannews.csv")


# In[ ]:





# In[ ]:





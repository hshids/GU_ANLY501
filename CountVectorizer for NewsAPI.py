#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#read CSV file to a dataframe
rawdata_DF = pd.read_csv('/Users/hanjingshi/Desktop/GGT/Newsapi_data.csv')
rawdata_DF.head()


# In[4]:


rawdata_DF = rawdata_DF.drop(['Unnamed: 0','publishedAt','url','urlToImage','source.id','source.name'], axis=1)
rawdata_DF = rawdata_DF.fillna(value={'author':'Unknown'})


# In[5]:


rawdata_DF.to_csv('Cleaned_News_Data.csv', index=False)


# In[6]:


MyCV_content=CountVectorizer(input='content',
                        stop_words='english',
                        max_features=100
                        )


# In[8]:


My_DTM = MyCV_content.fit_transform(rawdata_DF['title'])
ColNames = MyCV_content.get_feature_names()
print("The vocab is: ", ColNames, "\n\n")
rawdata_DF_content = pd.DataFrame(My_DTM.toarray(), columns=ColNames)
print(rawdata_DF_content)

rawdata_DF_content.insert(loc=0, column='LABEL', value=rawdata_DF['title'])
print(rawdata_DF_content)

for voc in ColNames:
    if voc.isdigit() == True:
        rawdata_DF_content = rawdata_DF_content.drop(voc, axis=1)

rawdata_DF_content.to_csv('CV_News.csv', index=False)


# In[ ]:





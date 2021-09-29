#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import numpy as np
import pandas as pd
import os
import re

url = ('https://newsapi.org/v2/everything?'
       'q=COVID impact technology companies OR COVID marketing or COVID impact technology companies &'
       'from=2021-09-29&'
       'sortBy=popularity&'
       'apiKey=5a712eed25e441de8079b2c5326b4b61')

##response = requests.get(url)

print(url)
response2 = requests.get(url)
jsontxt2 = response2.json()
print(jsontxt2, "\n")


# In[ ]:





# In[2]:


for i in jsontxt2['articles']:
    print(i['title'])


# In[ ]:





# In[4]:



from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[5]:


# Create an empty string
text_combined = ''
# Loop through all the headlines and add them to 'text_combined' 
for i in jsontxt2['articles']:
    text_combined += i['title'] + ' ' # add a space after every headline, so the first and last words are not glued together
# Print the first 300 characters to screen for inspection
print(text_combined[0:300])


# In[6]:


wordcloud = WordCloud(max_font_size=40).generate(text_combined)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





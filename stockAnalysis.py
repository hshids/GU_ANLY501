#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install yfinance


# In[6]:


import yfinance as yf

# Get the data for the stock ZOOM
zoom = yf.download('ZM','2019-01-01','2021-09-12')
zoom.to_csv('~/Desktop/CLEAN/zoom.csv')
# Import the plotting library
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Plot the close price of the AAPL
zoom['Adj Close'].plot()
plt.show()


# In[7]:


import yfinance as yf


# Get the data for the stock AMAZON
amazon = yf.download('AMZN','2019-01-01','2021-09-12')
amazon.to_csv('~/Desktop/CLEAN/amazon.csv')
# Import the plotting library
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Plot the close price of the AAPL
amazon['Adj Close'].plot()
plt.show()


# In[8]:


import yfinance as yf

# Get the data for the stock Walmart
wmt = yf.download('WMT','2019-01-01','2021-09-12')
wmt.to_csv('~/Desktop/CLEAN/wmt.csv')
# Import the plotting library
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Plot the close price of the AAPL
wmt['Adj Close'].plot()
plt.show()


# In[10]:


import yfinance as yf

# Get the data for the stock Costco
cost = yf.download('COST','2019-01-01','2021-09-12')
cost.to_csv('~/Desktop/CLEAN/cost.csv')
# Import the plotting library
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Plot the close price of the AAPL
cost['Adj Close'].plot()
plt.show()


# In[ ]:


Citation
Shah, I. (2021, May 3). Historical Stock Price Data in Python - Towards Data Science. Medium. https://towardsdatascience.com/historical-stock-price-data-in-python-a0b6dc826836


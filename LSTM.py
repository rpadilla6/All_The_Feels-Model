
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing import sequence
from keras.datasets import imdb
import re
from gensim.models import Word2Vec


# In[3]:


pip install gensim


# In[1]:


import sys 
print(sys.path)


# In[2]:


data = pd.read_csv('dev.txt', index_col=0, delimiter='\t')


# In[ ]:


data2 = pd.read_csv('train.txt', index_col=0, delimiter='\t')


# In[ ]:


data3 = pd.read_csv('test.txt', index_col=0, delimiter='\t')


# In[3]:


data.head()


# In[4]:


feature_names  = data.columns


# In[5]:


feature_names


# In[6]:


data.iloc[:,0] = data.iloc[:,0].apply(lambda x: re.sub('@\w+', '', x))


# In[7]:


data.iloc[:,0] = data.iloc[:,0].apply(lambda x: re.sub('[^A-Za-z1-9!?, ]', '', x))


# In[10]:


data.iloc[:,0] = data.iloc[:,0].apply(lambda x: x.lower())


# In[11]:


data.head()


# In[21]:


vectorize = W


# In[ ]:


X = full['Tweet']
Y = full['anger']


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[ ]:


embedded_vector_length = 32
model = Sequential()
model.add(Embedding(1000, embedded_vector_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()


# In[ ]:


model.fit(x_train, y_train, epochs=5, batch_size=32)


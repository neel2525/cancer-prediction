#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[34]:


dataset = pd.read_csv('train.csv')
X = dataset.iloc[:,1:10].values
y = dataset.iloc[:,10].values


# In[3]:


dataset


# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[36]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[37]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[38]:


y_pred = classifier.predict(X_test)


# In[39]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[40]:


dataset1 = pd.read_csv('test.csv')
X1 = dataset1.iloc[:,1:10].values


# In[41]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1 = sc.fit_transform(X1)


# In[42]:


y1 = classifier.predict(X1)


# In[43]:


dk=pd.DataFrame(y1)


# In[44]:


y1


# In[19]:


dt=pd.read_csv('r1.csv')


# In[45]:


dv=pd.concat([dt['Id'],dk],axis=1)
dv.columns=['Id','class']
dv.to_csv('kaggle2.csv',index=False)


# In[23]:


dv=pd.concat([dt[:Id],dk],axis=1)
dv.to_csv('kaggle.csv',index=false)


# In[ ]:





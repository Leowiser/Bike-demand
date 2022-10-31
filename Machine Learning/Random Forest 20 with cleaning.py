#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Final clustered data/Clustered_weather_data20.csv")
df.head()


# In[3]:


last_column = df.pop('demand')
df.insert(16, 'demand', last_column)
df.head()


# In[4]:


df.drop('date', axis=1, inplace=True)


# In[5]:


df['demand'].max()


# In[6]:


df['demand'].quantile([.95])


# In[7]:


index_names = df[ df['demand'] > 265].index

df.drop(index_names, inplace = True)


# In[8]:


params = {
    'bootstrap':[True, False],
    'max_depth': [10, 12, 14, 16],
    'max_features':[2, 3, 4, 5],
    'min_sample_leaf':[4, 8, 12],
    'n_estimators':[50, 100, 150]
}


# In[9]:


labels = np.array(df['demand'])# Remove the labels from the features
features= df.drop('demand', axis = 1)# Saving feature names for later use
feature_list = list(df.columns)# Convert to numpy array
features = np.array(df)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# In[10]:


features


# In[12]:


df_sample = df.sample(n=1000)


# In[13]:


Xs = df.iloc[:, :-1]
ys = df.iloc[:, -1]


# In[14]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split# Split the data into training and testing sets
Xs_cluster_train, Xs_cluster_test, ys_cluster_train, ys_cluster_test = train_test_split(Xs, ys,  test_size=0.2, random_state=123)


# In[10]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split# Split the data into training and testing sets
X_cluster_train, X_cluster_test, y_cluster_train, y_cluster_test = train_test_split(X, y,  test_size=0.4, random_state=123)


# In[11]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


# In[17]:


rfr = RandomForestRegressor()


# In[18]:


param_dist = {'bootstrap': [True, False],
               'max_depth': [30, 40, 50, 60, 70, 80],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'n_estimators': [100, 150, 200]}


# In[19]:



# Run randomized search
random_search = RandomizedSearchCV(estimator=rfr, 
                                   param_distributions=param_dist,
                                   n_iter=50,
                                   n_jobs=-1)


# In[ ]:


random_search.fit(Xs_cluster_train, ys_cluster_train)


# In[ ]:


random_search.best_params_


# In[ ]:


random_search.best_estimator_


# In[12]:


# Import the model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(criterion="squared_error", max_depth=60, min_samples_leaf=2, min_samples_split=2,n_estimators=150)


# In[13]:


import time
start = time.time()
rf.fit(X_cluster_train, y_cluster_train)
end = time.time()
print(f"it takes {end - start} seconds to fit the model.")


# In[14]:


predictions_cluster = rf.predict(X_cluster_test)
predictions_cluster


# In[15]:


y_cluster_test


# In[16]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_cluster_test, predictions_cluster))
print("RMSE: %f" % (rmse))


# In[17]:


from sklearn.metrics import r2_score
r2_score(y_cluster_test, predictions_cluster)


# In[18]:


x = y_cluster_test - predictions_cluster


# In[19]:


x.mean()


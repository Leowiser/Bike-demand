#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# First libraries and the data is imported.


# In[3]:


df = pd.read_csv("C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Final clustered data/Clustered_weather_data_supply.csv")
df.info()


# In[4]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error


# In[5]:


# For later simplification, the demand is switched to the last column


# In[6]:


last_column = df.pop('supply')
df.insert(16, 'supply', last_column)
df.head()


# In[ ]:





# In[7]:


# As it is not necessary for the prediction, date and year column is droped.


# In[8]:


df.drop('date', axis=1, inplace=True)


# In[9]:


df.drop('year', axis = 1, inplace = True)


# In[10]:


# Correlation is between the variables is depicted in a correlation chart.


# In[11]:


corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (12,12))
f = sns.heatmap(df[top_corr_features].corr(), annot = True)


# In[12]:


# Now outliers are controlled for.
# The highest demand value is investigated.


# In[13]:


df['supply'].max()


# In[14]:


# Now it is tested what the 99.5% quantil value is. 


# In[16]:


df['supply'].quantile([.995])


# In[16]:


# as the value is half as the maximum value. These extreme outliers are droped as they most likely depict certain situation that are very unique.


# In[18]:


index_names = df[ df['supply'] > 567].index

df.drop(index_names, inplace = True)


# In[19]:


len(pd.unique(df['cluster_dtw']))


# In[19]:


# There still are 29 clusters included in the data.


# In[20]:


# Now the Data is used for a prediction.


# In[21]:


# First the data is split into training and test data.


# In[20]:


from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


# In[22]:


# First a Hyper Parameter Optimization for the XGBoost algorithm is done.

params={
 "learning_rate"    : [0.20, 0.25, 0.30, 0.35, 0.4] ,
 "max_depth"        : [10, 12, 14, 16],
 "min_child_weight" : [5, 7, 9, 11],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.5 , 0.7, 0.8 , 0.9, 1.0]
    
}


# In[23]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost


# In[24]:


# for the optimization Randomized Search CV is used, as the Data set used is large.


# In[25]:


regressor=xgboost.XGBRegressor()


# In[26]:


random_search=RandomizedSearchCV(regressor,param_distributions=params,n_iter=10,scoring='neg_root_mean_squared_error',n_jobs=-1,cv=2,verbose=3)


# In[27]:


random_search.fit(X_train,y_train)


# In[28]:


random_search.best_estimator_


# In[29]:


# This is the best estimator.


# In[30]:


random_search.best_params_


# In[31]:


# The result of the optimazation is used for the XGBoost algorithm.


# In[32]:


data_dmatrix = xgb.DMatrix(data=X,label=y)


# In[34]:


#"objective":"reg:linear"
xg_reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.7, enable_categorical=False,
             gamma=0.3, gpu_id=-1, importance_type=None,
             interaction_constraints='', learning_rate=0.2, max_delta_step=0,
             max_depth=14, min_child_weight=11,
             monotone_constraints='()', n_estimators=100, n_jobs=8,
             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None, seed=140)


# In[35]:


import time
start = time.time()
xg_reg.fit(X_train,y_train)
end = time.time()
print(f"it takes {end - start} seconds to fit the model.")


# In[36]:


preds = xg_reg.predict(X_test)
preds


# In[37]:


y_test


# In[38]:


rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


# In[39]:


# The final resulting rmse value is 22.75

# The rsquared value is calculated


# In[40]:


from sklearn.metrics import r2_score
r2_score(y_test, preds)


# In[41]:


x = y_test - preds
x.mean()


# In[42]:


# the r2 score is 0.95 which is already a good result.


# In[43]:


# Finally it is plotted which the most important features for the predictions are.


# In[44]:


xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()


# In[ ]:


# It turns out, that the most important features are by far the cluster and the hour the bike is rented.


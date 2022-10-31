
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


import numpy as np
import scipy
import sklearn
from tslearn.clustering import TimeSeriesKMeans

#Load combined data
df = pd.read_csv('C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/k_cluster.csv')

# drop every weekend day as they produce completely different results.
df.drop(df[df['day_of_week'] > 4].index)


# cluster each in the single clusters
# cluster 0
filtered_values_0 = np.where((df['cluster']==4))
print(filtered_values_0)
df_0 = df.copy()
df_0 = df_0.loc[filtered_values_0]
df_0.info()

steps = 24
Tage = len(pd.unique(df_0['day']))

#Anzahl an Zeit elementen
Zeit = Tage * steps

stations = len(pd.unique(df_0['station_id']))


df_0 = df_0.groupby(["station_id",'hour']).agg({'supply':'mean'})

df_0.reset_index(inplace=True)

df_0_piv = df_0.pivot(index='station_id', columns='hour', values='supply')

model = TimeSeriesKMeans(n_clusters=4, metric="softdtw", max_iter=10)
model.fit(df_0_piv)

df_0_piv['cluster_dtw'] = model.labels_
df_0_piv.head()

list_0 = df_0_piv['cluster_dtw'].tolist()



def dtw(x, l):
    n = Zeit
    k = []
    j = 0
    while j < x:
        k += [l[j]] * n
        j += 1
    return k


d = pd.Series(dtw(stations, list_0), name = 'cluster_dtw')

df_0_dtw = pd.concat([d], axis = 1)
df_0_dtw.head()

cluster_0_dtw = df_0_dtw['cluster_dtw']
df_0.insert(1, "cluster_dtw", cluster_0_dtw)

df_0.head()
pd.unique(df_0['cluster_dtw'])


df_0['cluster_dtw'] = df_0['cluster_dtw'] + 40

f =sns.lineplot(x="hour", y="supply", hue="cluster_dtw",data=df_0, legend='full')
plt.show()

df_0.drop('hour', axis=1, inplace=True)
df_0.drop('supply', axis=1, inplace=True)
len(pd.unique(df_0['station_id']))
df = df_0.drop_duplicates()

df.to_csv(r'C:\Users\leonw\Documents\Studium\6.Semester\Data Literacy\Hausarbeit\bike data\Clustered data\cluster_5_s.csv', index = False)


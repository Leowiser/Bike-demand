
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


import numpy as np

# cycling data
c1 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_1.csv"
c2 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_2.csv"
c3 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_3.csv"
c4 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_4.csv"
c5 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_5.csv"
c6 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_6.csv"
c7 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_7.csv"
c8 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_8.csv"
c9 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_9.csv"
c10 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_10.csv"
c11 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_11.csv"
c12 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_12.csv"
c13 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_13.csv"
c14 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_14.csv"
c15 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_15.csv"


Comb_data = pd.concat(map(pd.read_csv, [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15]), ignore_index=True)
Comb_data.info()
len(c1)

Comb_data.drop('hour', axis=1, inplace=True)
Comb_data.drop('balance', axis=1, inplace=True)
Comb_data['cluster_dtw'].astype(float)

df_combined = Comb_data.drop_duplicates()
df_combined.info()

df_location_cluster = pd.read_csv('C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/k_cluster.csv')

df_bike = df_location_cluster.merge(df_combined,on=['station_id'], how = 'left')
df_bike.head()
len(pd.unique(df_bike['cluster_dtw']))
j
bike = df_bike.groupby(["cluster_dtw",'month','day','hour']).agg({'demand':'sum',"supply":'sum','balance':'sum'})
bike.info()
bike.head(60)



import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


import numpy as np

# cycling data
c1 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_1_d.csv"
c2 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_2_d.csv"
c3 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_3_d.csv"
c4 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_4_d.csv"
c5 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_5_d.csv"
c6 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_6_d.csv"
c7 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_7_d.csv"
c8 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_8_d.csv"
c9 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_9_d.csv"
c10 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_10_d.csv"
c11 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_11_d.csv"
c12 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_12_d.csv"
c13 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_13_d.csv"
c14 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_14_d.csv"
c15 = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/cluster_15_d.csv"


Comb_data = pd.concat(map(pd.read_csv, [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15]), ignore_index=True)
Comb_data.info()

Comb_data['cluster_dtw'].astype(float)

(pd.unique(Comb_data['cluster_dtw']))


df_location_cluster = pd.read_csv('C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/k_cluster.csv')

df_bike = df_location_cluster.merge(Comb_data,on=['station_id'], how = 'left')
df_bike.info()
len(pd.unique(df_bike['cluster_dtw']))

bike = df_bike.groupby(["cluster_dtw",'month','day','hour']).agg({'demand':'sum'})


bike.reset_index(inplace=True)
bike['year'] = 2019

cols = ['year','month','day']
bike_clustered = bike.copy()
bike_clustered['year']= bike_clustered['year'].astype(int)
bike_clustered['month']= bike_clustered['month'].astype(int)
bike_clustered['day']= bike_clustered['day'].astype(int)
bike_clustered['date'] = bike_clustered[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis = 'columns')
bike_clustered['date'] = pd.to_datetime(bike_clustered['date'])
len(pd.unique(bike_clustered['date']))

weather = pd.read_csv('C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/weather_nyc.csv')
weather['date'] = pd.to_datetime(weather['date'])
weather.info()

bike_clustered_weather = bike_clustered.merge(weather,on=['date'], how = 'left')
bike_clustered_weather.info()
len(pd.unique(bike_clustered_weather['cluster_dtw']))
bike_clustered_weather.to_csv(r'C:\Users\leonw\Documents\Studium\6.Semester\Data Literacy\Hausarbeit\bike data\Final clustered data\Clustered_weather_data.csv', index = False)
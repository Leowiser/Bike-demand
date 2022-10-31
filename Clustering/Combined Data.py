# New York Bicycle prediction
# weather data set.

import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


import numpy as np

# cycling data
January = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Combined data per month/Bike_data_Jan_h.csv"
Feburary = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Combined data per month/Bike_data_Feb_h.csv"
March = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Combined data per month/Bike_data_Mar_h.csv"
April = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Combined data per month/Bike_data_Apr_h.csv"
May = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Combined data per month/Bike_data_May_h.csv"
June = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Combined data per month/Bike_data_Jun_h.csv"
July = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Combined data per month/Bike_data_Jul_h.csv"
August = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Combined data per month/Bike_data_Aug_h.csv"
September = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Combined data per month/Bike_data_Sep_h.csv"
October = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Combined data per month/Bike_data_Oct_h.csv"
November = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Combined data per month/Bike_data_Nov_h.csv"
December = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Combined data per month/Bike_data_Dec_h.csv"


Comb_data = pd.concat(map(pd.read_csv, [January, Feburary, March, April, May, June, July, August, September, October, November, December]), ignore_index=True)

print(Comb_data.head())
print(len(Comb_data))
print(Comb_data.describe())
print(len(pd.unique(Comb_data['station_id'])))

# Data engineering
# Holidays might have an impact on the demand of bikes at certain stations
# dates of holidays: 01.01, 21.01, 12.02, 18.02, 27.05, 04.07, 02.09,14.10,11.11, 28.11, 25.12

def holidays(x):
    holiday = []
    for i in x:
        if i == '2019-01-01':
            holiday.append(1)
        elif i == '2019-01-21':
            holiday.append(1)
        elif i == '2019-02-12':
            holiday.append(1)
        elif i == '2019-02-18':
            holiday.append(1)
        elif i == '2019-05-27':
            holiday.append(1)
        elif i == '2019-07-04':
            holiday.append(1)
        elif i == '2019-09-02':
            holiday.append(1)
        elif i == '2019-10-14':
            holiday.append(1)
        elif i == '2019-11-11':
            holiday.append(1)
        elif i == '2019-11-28':
            holiday.append(1)
        elif i == '2019-12-25':
            holiday.append(1)
        else:
            holiday.append(0)
    return holiday

is_holiday = holidays(Comb_data['date'])

Comb_data['is_holiday'] = is_holiday
#is_holiday is a boolean taking the value of 1 when the day is an official holiday

Comb_data.info()

# K nearest neighbour clustering

data = Comb_data.copy()

data['demand'] = data['demand'].fillna(0)
data.drop('demand', axis=1, inplace=True)
data.drop('month', axis=1, inplace=True)

data.drop('day', axis = 1, inplace = True)
data.drop('hour', axis = 1, inplace = True)
data.drop('is_holiday', axis = 1, inplace = True)
data.drop('supply', axis = 1, inplace = True)
data.drop('date', axis = 1, inplace = True)
data.drop('day_of_week', axis = 1, inplace = True)
data.drop('balance', axis = 1, inplace = True)
data.head()
data = data.drop_duplicates(subset = 'station_id')

data.info()
data.set_index(data['station_id'], drop=True, inplace=True)
data.head()
data.drop('station_id', axis = 1, inplace = True)
data.head()
print(data.isnull().sum())

X = data.to_numpy()
kmeans = KMeans(n_clusters=15, random_state=0).fit(X)

data['cluster'] = kmeans.labels_
data.head()
plt.figure(figsize=(12, 12))
g =sns.scatterplot(x="longitude", y="latitude", hue="cluster",data=data,palette=['green','orange','brown','dodgerblue','red','pink','blanchedalmond','black','beige','lime','peru','lightcyan','yellowgreen','salmon','orangered'],legend='full')

unite = data.copy()
unite.reset_index(inplace=True)
unite.drop('longitude', axis=1, inplace=True)
unite.drop('latitude', axis=1, inplace=True)
unite.info()
unite.head(100)

Comb_data = Comb_data.merge(unite,on=['station_id'], how = 'left')
Comb_data.head(50)

Comb_data.to_csv(r'C:\Users\leonw\Documents\Studium\6.Semester\Data Literacy\Hausarbeit\bike data\Clustered data\k_cluster.csv', index = False)
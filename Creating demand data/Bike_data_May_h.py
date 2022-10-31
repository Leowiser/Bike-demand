import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt


bike = pd.read_csv('C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/201905-citibike-tripdata.csv')


bike.rename(columns = {'start station latitude':'start_lat', 'start station longitude':'start_lon', 'end station latitude':'end_lat', 'end station longitude':'end_lon'}, inplace = True)


bike = bike[(bike['start_lon']>-74.2) & (bike['start_lon']<(-73.91))]
bike = bike[(bike['start_lat']>40.66) & (bike['start_lat']<40.825)]


bike= bike.astype({"starttime":str})

bike['starttime'] = pd.to_datetime(bike['starttime'])
bike['stoptime'] = pd.to_datetime(bike['stoptime'])

# new columns for the different dates
bike['year'] = bike['starttime'].dt.year
bike['month'] = bike['starttime'].dt.month
bike['day_start'] = bike['starttime'].dt.day
bike['day_of_week_start'] = bike['starttime'].dt.dayofweek
bike['hour_start'] = bike['starttime'].dt.hour

bike['month_end'] = bike['stoptime'].dt.month
bike['day_end'] = bike['stoptime'].dt.day
bike['day_of_week_end'] = bike['stoptime'].dt.dayofweek
bike['hour_end'] = bike['stoptime'].dt.hour

print(bike.isnull().sum())

# There are no missing values in the bike.

# create a list of the conditions
conditions = [
    (bike['hour_start'] <= 3),
    (bike['hour_start'] > 3) & (bike['hour_start'] <= 6),
    (bike['hour_start'] > 6) & (bike['hour_start'] <= 9),
    (bike['hour_start'] > 9) & (bike['hour_start'] <= 12),
    (bike['hour_start'] > 12) & (bike['hour_start'] <= 15),
    (bike['hour_start'] > 15) & (bike['hour_start'] <= 18),
    (bike['hour_start'] > 18) & (bike['hour_start'] <= 21),
    (bike['hour_start'] > 21) & (bike['hour_start'] <= 24)
    ]

# create a list of the values we want to assign for each condition
values = [0, 1, 2, 3, 4, 5, 6, 7]

# create a new column and use np.select to assign values to it using our lists as arguments
bike['timesteps'] = np.select(conditions, values)

# display updated DataFrame
bike.info()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



# Creating Data Frames that include the demand and supply at each timestep and station.

bike['supply'] = 1

bike['demand'] = 1

df_demand = bike.groupby(["start station id",'day_start','hour_start']).agg({"demand":"count","month":'mean'})

df_demand.reset_index(inplace=True)
df_demand.rename(columns = {'start station id':'station_id', 'hour_start':'hour', 'day_start':'day'}, inplace = True)

df_supply = bike.groupby(['end station id','day_end','hour_end']).agg({"supply":"count", "month":'mean'})
df_supply.reset_index(inplace=True)
df_supply.rename(columns = {'end station id':'station_id', 'hour_end':'hour', 'day_end':'day'}, inplace = True)

# ## Connecting the Data Frames
#
# This is done by creating an empty Data Frame including all possible solutions.
m = 5

timestepsteps = 8
steps = 24


#Anzahl unterschiedlicher stationen
stations = len(pd.unique(bike['start station id']))

#Länge der Listen
Tage = len(pd.unique(bike['day_start']))
Länge = Tage*steps*stations

#Anzahl an Zeit elementen
Zeit = Tage * steps

#List of station ids
stationid = pd.unique(df_demand['station_id'])

def timesteps(x):
    tsemp = []
    i = 0
    while i < x:
        tsemp.append(0)
        tsemp.append(1)
        tsemp.append(2)
        tsemp.append(3)
        tsemp.append(4)
        tsemp.append(5)
        tsemp.append(6)
        tsemp.append(7)
        i += 8
    return tsemp

def hour(x):
    tsemp = []
    i = 0
    while i < x:
        tsemp.append(0)
        tsemp.append(1)
        tsemp.append(2)
        tsemp.append(3)
        tsemp.append(4)
        tsemp.append(5)
        tsemp.append(6)
        tsemp.append(7)
        tsemp.append(8)
        tsemp.append(9)
        tsemp.append(10)
        tsemp.append(11)
        tsemp.append(12)
        tsemp.append(13)
        tsemp.append(14)
        tsemp.append(15)
        tsemp.append(16)
        tsemp.append(17)
        tsemp.append(18)
        tsemp.append(19)
        tsemp.append(20)
        tsemp.append(21)
        tsemp.append(22)
        tsemp.append(23)

        i += 24
    return tsemp

def ids(x, y):
    ids = []
    j = 0
    n = Zeit
    while j < x:
        ids += [y[j]]*n
        j += 1
    return ids

def lat(x, y):
    lat = []
    j = 0
    n = Zeit
    while j < x:
        lat += [y[j]]*n
        j += 1
    return lat

def lon(x, y):
    lon = []
    j = 0
    n = Zeit
    while j < x:
        lon += [y[j]]*n
        j += 1
    return lon

def days(z):
    days = []
    n = steps
    j = 0
    while j < z:
        days += [1]*n
        days += [2]*n
        days += [3]*n
        days += [4]*n
        days += [5]*n
        days += [6]*n
        days += [7]*n
        days += [8]*n
        days += [9]*n
        days += [10]*n
        days += [11]*n
        days += [12]*n
        days += [13]*n
        days += [14]*n
        days += [15]*n
        days += [16]*n
        days += [17]*n
        days += [18]*n
        days += [19]*n
        days += [20]*n
        days += [21]*n
        days += [22]*n
        days += [23]*n
        days += [24]*n
        days += [25]*n
        days += [26]*n
        days += [27]*n
        days += [28]*n
        days += [29]*n
        days += [30]*n
        days += [31]*n
        j += 1
    return days

Time = pd.Series(timesteps(Länge), name = 'timesteps')
Hours = pd.Series(hour(Länge), name = 'hour')


station_id = pd.Series(ids(stations, stationid), name = 'station_id')


latitude = pd.Series(lat(stations, pd.unique(bike['start_lat'])), name = 'latitude')
longitude = pd.Series(lon(stations, pd.unique(bike['start_lon'])), name = 'longitude')


days = pd.Series(days(stations), name = 'day')


df_empty = pd.concat([station_id, Hours, days, latitude, longitude], axis = 1)
df_empty['month'] = m
df_empty.head()

df_conect = df_empty.merge(df_demand,on=['hour','station_id','day','month'], how = 'left')


df_merged = df_conect.merge(df_supply, on=['hour','station_id','day','month'], how = 'left')


date = pd.date_range(start='05-01-2019', end='05-31-2019')
time = pd.DataFrame(date)
time['day'] = time[0].dt.day

# Now the data frame is merged with the dates.
# Date of the week are also connected.
# Also, the NaN of demand and supply are equal to zero and are thus replaced with it.
#

df_bike = df_merged.merge(time,on=['day'], how = 'left')
df_bike.rename(columns = {0:'date'}, inplace = True)
df_bike['day_of_week'] = df_bike['date'].dt.dayofweek
df_bike['demand'] = df_bike['demand'].fillna(0)
df_bike['supply'] = df_bike['supply'].fillna(0)


df_bike['balance'] = df_bike['supply'] - df_bike['demand']

df_bike.to_csv(r'C:\Users\leonw\Documents\Studium\6.Semester\Data Literacy\Hausarbeit\bike data\Combined data per month\Bike_data_May_h.csv', index = False)



import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# cycling data
January = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/201901-citibike-tripdata.csv"
Feburary = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/201902-citibike-tripdata.csv"
March = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/201903-citibike-tripdata.csv"
April = "C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/201904-citibike-tripdata.csv"


bike = pd.concat(map(pd.read_csv, [January, Feburary, March, April]), ignore_index=True)

bike = pd.read_csv('C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/201901-citibike-tripdata.csv')

bike['starttime'] = pd.to_datetime(bike['starttime'])
bike['stoptime'] = pd.to_datetime(bike['stoptime'])

bike['year'] = bike['starttime'].dt.year
bike['month'] = bike['starttime'].dt.month
bike['day_start'] = bike['starttime'].dt.day
bike['day_of_week_start'] = bike['starttime'].dt.dayofweek
bike['hour_start'] = bike['starttime'].dt.hour

bike['month_end'] = bike['stoptime'].dt.month
bike['day_end'] = bike['stoptime'].dt.day
bike['day_of_week_end'] = bike['stoptime'].dt.dayofweek
bike['hour_end'] = bike['stoptime'].dt.hour
bike_week = bike.copy()

bike_week.info()
bike_week.drop(bike_week[bike_week['day_of_week_start'] > 4].index)


sns.distplot(bike_week['hour_start'], kde = False)
plt.title('Bike demand by hour', fontsize=18)
plt.xlabel('Hours', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.show()


bike_weekend = bike.copy()



sns.distplot(bike_weekend['day_of_week_start'], kde = False)
plt.title('Bike demand by day', fontsize=18)
plt.xlabel('Days', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.show()

bike_weekend_p = bike.copy()

bike_weekend_p = bike_weekend_p.drop(bike_weekend_p[bike_week['day_of_week_start'] < 5].index)

sns.distplot(bike_weekend_p['hour_start'], kde = False, color = 'orange')
plt.title('Bike demand by day (weekend)', fontsize=18)
plt.xlabel('Days', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.show()
bike_weekend_p.head()

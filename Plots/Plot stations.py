
import matplotlib.pyplot as plt
import pandas as pd


bike = pd.read_csv('C:/Users/leonw/Documents/Studium/6.Semester/Data Literacy/Hausarbeit/bike data/Clustered data/k_cluster.csv')

bike = bike[(bike['longitude']>-74.2) & (bike['longitude']<(-73.91))]
bike = bike[(bike['latitude']>40.66) & (bike['latitude']<40.825)]
bike.info()

plt.figure(figsize=(12, 12))
plt.scatter(bike['longitude'], bike['latitude'], s = 100, c = 'red', marker = '.', cmap = 'bike stations', alpha = 0.8)
plt.title("Locations in NYC")
plt.grid(None)
plt.show()
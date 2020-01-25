import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Plots a bar graph showing the distribution of percantage gains per stock during the training

dataraw = pd.read_csv("logdata/LogdataTest.csv")
tickers = pd.read_csv("data/SP100tickers.txt").transpose()

#Taking the last 10% of data, dont want to include training data where epsilon has been high.
trainlen = len(dataraw) - (len(dataraw) * 0.9)

#If the data you want to plot happens to be less than 1000 rows, ignore this an use all the data
if len(dataraw) < 1000:
    trainlen = 0

datasub = dataraw.iloc[int(trainlen):,]


datasubgroup = datasub.groupby("Stocknr").mean()
datasubgroup["tickers"] = tickers.index[0:100].map(str)
datasubgroup = datasubgroup.sort_values("reward_pcr")
symbols = datasubgroup.tickers
print("Sum current port",sum(datasubgroup.current_port_sum))
print("Sum benchmark port", sum(datasubgroup.benchmark_port_sum))
print("Result", sum(datasubgroup.current_port_sum) - sum(datasubgroup.benchmark_port_sum))
#Y axis the mean percentage gain per stock, X axis the tickers
plt.figure(figsize=(20,15))
#plt.bar(symbols,datasubgroup.reward_pcr)
plt.bar(symbols,datasubgroup.current_port_sum - datasubgroup.benchmark_port_sum)
plt.xticks(rotation=90)
plt.show()


import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import numpy as np



#Latest log file
list_of_files = glob.glob('logdata/*') 
latest_file = str(max(list_of_files, key=os.path.getctime))
#latest_file = "logdata/LogdataTest_256x256.20c_RewSha-0.5_test_serie.csv"


#Plots a bar graph showing the distribution of percantage gains per stock during the training
dataraw = pd.read_csv(latest_file)
tickers = pd.read_csv("data/SP500-100tickers.txt", header = None).transpose()[0].tolist()[:dataraw.shape[0]-1]

#Taking the last 10% of data, dont want to include training data where epsilon has been high.
trainlen = len(dataraw) - (len(dataraw) * 0.1)

#If the data you want to plot happens to be less than 1000 rows, ignore this an use all the data
if len(dataraw) < 2500:
    trainlen = 0

datasub = dataraw.iloc[int(trainlen):,]
datasubgroup = datasub.groupby("Stocknr").mean()

#Debug
print("Latest file:",latest_file)
print(datasubgroup.shape)
print(len(tickers))



datasubgroup["tickers"] = tickers#.map(str)
datasubgroup = datasubgroup.sort_values("reward_pcr")
symbols = datasubgroup.tickers
print("Sum current port",sum(datasubgroup.current_port_sum))
print("Sum benchmark port", sum(datasubgroup.benchmark_port_sum))
print("Result", sum(datasubgroup.current_port_sum) - sum(datasubgroup.benchmark_port_sum))
#Y axis the mean percentage gain per stock, X axis the tickers
plt.figure(figsize=(20,5))
#plt.bar(symbols,datasubgroup.reward_pcr)
plt.bar(symbols,datasubgroup.current_port_sum - datasubgroup.benchmark_port_sum)
plt.xticks(rotation=90)
plt.show()


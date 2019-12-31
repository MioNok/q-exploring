
import pandas as pd
import numpy as np
import time


def fetchdowstockdata(aphkey, data = None ): # perhaps add apikey as a flag, returns the data in a pandas dataframe, columns = timestamp, open, high, low, close, volume. Data is daily.

     if data == None:
          #Read data from file. Transpose it, then to series and lastly to list.
          symbols = pd.read_csv("dowtickers.txt", header = None).transpose()[0].tolist()

          #Fetch data from api
          stockdata = pd.DataFrame()
          for symbol in symbols:
               stockdata_aph = pd.read_csv("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol="+ symbol +"&outputsize=full&apikey="+aphkey+"&datatype=csv")
               stockdata_aph["ticker"] = symbol
               stockdata = stockdata.append(stockdata_aph.iloc[0:2514,]) # Number of trading days during this decade. Edit this if for different time period, currently returns approx the last 10 years of data
               time.sleep(12) # Sleep because of the ratelimit on alphavantage
               print("fetched", symbol)

          #Save data       
          stockdata.to_csv("Dow2010-2019data.csv",index = None, header = True)
     
     else:
          #TODO:read from file.
          pass

     return stockdata



    
    
    

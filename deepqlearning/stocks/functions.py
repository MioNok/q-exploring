
import pandas as pd
import numpy as np
import time
import pandasql as ps


def fetchdowstockdata(aphkey): # perhaps add apikey as a flag, returns the data in a pandas dataframe, columns = timestamp, open, high, low, close, volume. Data is daily.

    #Read data from file. Transpose it, then to series and lastly to list.
    symbols = pd.read_csv("dowtickers.txt", header = None).transpose()[0].tolist()

    #Fetch data from api
    stockdata = pd.DataFrame()
    for symbol in symbols:
        stockdata_aph = pd.read_csv("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol="+ symbol +"&outputsize=full&apikey="+aphkey+"&datatype=csv")
        stockdata_aph["ticker"] = symbol
        stockdata = stockdata.append(stockdata_aph.iloc[1:2517,]) # Number of trading days during this decade. Edit this if for different time period, currently returns approx the last 10 years of data
        time.sleep(12) # Sleep because of the ratelimit on alphavantage
        print("fetched", symbol)

    #Save data       
    stockdata.to_csv("Dow2010-2019AdjustedData.csv",index = None, header = True)
     

    return stockdata

def preprocessdata(datafilename,tickerstxt,limit):
    
    #The data
    stoset_raw = pd.read_csv(datafilename)
    symbols = pd.read_csv(tickerstxt, header = None).transpose()[0].tolist()
    
    stockdata = [] # Un normalized data for keeping track of stock prices
    normalized_stockdata = [] # Normalized data for training
    
    progress_counter = 1
    for symbol in symbols:
        query = "SELECT timestamp, open, high, low, adjusted_close, volume FROM stoset_raw WHERE ticker = '"+symbol+"' LIMIT "+ str(limit)
        #query dataframe and reverse it so that the older data is first
        stoset = ps.sqldf(query).iloc[::-1] 
        
        #renaming the adjusted_close column to close to limit confusion in the rest of the program
        stoset.rename({"adjusted_close":"close"}, axis = 1, inplace = True) 
        
        #remove the timestamp for the version of the data that is normalized for training
        stoset_notimestamp = stoset.iloc[:,1:6] 
        
    
        #Normalizing using minmax normalzation
        normalized_stoset =(stoset_notimestamp-stoset_notimestamp.min())/(stoset_notimestamp.max()-stoset_notimestamp.min())
        
        #Append data to lists
        stockdata.append(stoset)
        normalized_stockdata.append(normalized_stoset)
        
        #Helpful profgress counter.
        print("Preprocessing stocks from data",progress_counter,"/",len(symbols))
        progress_counter += 1
        
    return stockdata, normalized_stockdata





    
    
    

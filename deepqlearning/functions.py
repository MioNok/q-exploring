
import pandas as pd
import numpy as np
import time
import pandasql as ps
import os
import matplotlib.pyplot as plt


#Logdata dataframe is saved here. Probably faster to have it globally than reading and writing to a large file.


def fetchstockdata(aphkey,test,ticker, tickerfile): # perhaps add apikey as a flag, returns the data in a pandas dataframe, columns = timestamp, open, high, low, close, volume. Data is daily.

    #If we are not running the test script
    if not test:
        #Read data from file. Transpose it, then to series and lastly to list.
        symbols = pd.read_csv(tickerfile, header = None).transpose()[0].tolist()#[150:]

        #Fetch data from api
        counter = 0
        stockdata = pd.DataFrame()
        for symbol in symbols:
            stockdata_aph = pd.read_csv("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol="+ symbol +"&outputsize=full&apikey="+aphkey+"&datatype=csv")
            stockdata_aph["ticker"] = symbol
            stockdata = stockdata.append(stockdata_aph) # Number of trading days during this decade. 
            time.sleep(12) # Sleep because of the ratelimit on alphavantage
            print("fetched", symbol, counter)
            counter +=1

            if counter == 150:
                break

        # Create testdata folder
        if not os.path.isdir('data'):
            os.makedirs('data')

        #Save data       
        stockdata.to_csv("data/SP500-100-All-time_1-150_data.csv",index = None, header = True)
     
    else:
        #Fetch data from api
        stockdata = pd.DataFrame()
        stockdata_aph = pd.read_csv("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol="+ ticker +"&outputsize=full&apikey="+aphkey+"&datatype=csv")
        stockdata_aph["ticker"] = ticker
        stockdata = stockdata.append(stockdata_aph.iloc[1:2517,]) # Number of trading days during this decade. Edit this if for different time period, currently returns approx the last 10 years of data
        print("fetched", ticker)

        # Create testdata folder
        if not os.path.isdir('testdata'):
            os.makedirs('testdata')

        #Save data       
        stockdata.to_csv("testdata/testdata.csv",index = None, header = True)

        #Make file for ticker
        f= open("testdata/testticker.txt","w+")
        f.write(ticker)
        f.close() 

    return stockdata

def preprocessdata(datafilename,tickerstxt,limit, stocklimit,numcandles,skipstock = 0, offset = 0, fulltest =False):
    
    #The data
    stoset_raw = pd.read_csv(datafilename)
    
    #Read tickers, turn to list, an limit acording to stocklimit
    symbols = pd.read_csv(tickerstxt, header = None).transpose()[0].tolist()[skipstock:stocklimit+skipstock]
    print(symbols)
    stockdata = [] # Un normalized data for keeping track of stock prices
    normalized_stockdata = [] # Normalized data for training
    
    progress_counter = 1
    preprocessed_stocks = 0        
    for symbol in symbols:
        query = "SELECT timestamp, open, high, low, adjusted_close, volume FROM stoset_raw WHERE ticker = '"+symbol+"' LIMIT "+ str(limit)+ " OFFSET "+ str(offset)
        #query dataframe and reverse it so that the older data is first
        stoset = ps.sqldf(query).iloc[::-1]

        if stoset.shape[0] < numcandles:
            print("Not enough data for", symbol)
            progress_counter += 1 
            continue

        if stoset.shape[0] < limit and fulltest == True:
            print("Not enough data for full test", symbol)
            progress_counter += 1 
            continue  

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
        
        #Print first and last date
        if progress_counter == 1:
            print("From:",stoset["timestamp"].iloc[0])
            print("To:",stoset["timestamp"].iloc[-1])
        print("Number of days:", stoset.shape[0])
        preprocessed_stocks += 1
        progress_counter += 1
        
    return stockdata, normalized_stockdata,preprocessed_stocks


def simplestrat(state,settings):

    #Simple trading stategy. If the algo cannot beat this its useless.
    pandas_state = pd.DataFrame(state)
    close_start = pandas_state.iloc[0,3]
    close_end = pandas_state.iloc[settings["Number_of_candles"]-1,3]

    #If the stock has closed 
    if close_start < close_end:
        action = 2
    else:
        action = 0

    return action


def appendlogdata(stocknr,episode,reward, current_port_sum, benchmark_port_sum, reward_pcr,logdatafile,tickers):

    #Logging the data and writing it to a csv 
    logdict = {"Stocknr": int(stocknr),
            "Ticker": tickers[int(stocknr)],
            "Episode": int(episode),
            "Reward": float(reward),
            "current_port_sum": int(current_port_sum),
            "benchmark_port_sum": int(benchmark_port_sum),
            "reward_pcr": float(reward_pcr)}

    logdatafile = logdatafile.append(logdict, ignore_index = True)
    return logdatafile


def appendlogdataporttest(stocknr,step, current_port_sum, benchmark_port_sum,logdatafile,tickers,action):

    #Logging the data and writing it to a csv 
    logdict = {
                #"Stocknr": int(stocknr),
                #"Ticker": tickers[int(stocknr)],
                "Step": int(step),
                "current_port_sum": int(current_port_sum),
                "benchmark_port_sum": int(benchmark_port_sum),
                "action": int(action)}

    logdatafile = logdatafile.append(logdict, ignore_index = True)
    return logdatafile

def writelogdata(logdatafile, settings, porttest = False):
    # Create testdata folder
    if not os.path.isdir('logdata'):
        os.makedirs('logdata')

    if porttest == True:
        #Update the file 
        logdatafile.to_csv("logdata/Port_test-Logdata"+ settings["Model_name"]+settings["Model_type"]+".csv")    
    else:
        #Update the file 
        logdatafile.to_csv("logdata/Logdata"+ settings["Model_name"]+settings["Model_type"]+".csv")    


#Plotting the results from the portfolio data test.
def plotporttestdata(model_name,model_type):
    portdata = pd.read_csv("logdata/Port_test-Logdata"+ model_name+model_type+".csv")

    algo_port_sum_list = []
    benchmark_port_sum_list = []

    current_step = 0
    temp_sum_algo = 0
    temp_sum_benchmark = 0
    for row in portdata.iterrows():
        if current_step == row[1].Step:
            temp_sum_algo += row[1].current_port_sum
            temp_sum_benchmark += row[1].benchmark_port_sum

            
        else:
            benchmark_port_sum_list.append(temp_sum_benchmark)
            algo_port_sum_list.append(temp_sum_algo)
            temp_sum_algo = row[1].current_port_sum
            temp_sum_benchmark = row[1].benchmark_port_sum
            current_step +=1

    plt.figure(figsize=(15,10))
    plt.plot(algo_port_sum_list, label = "Reinforced-Algo portfolio")
    plt.plot(benchmark_port_sum_list, label = "Benchmark portfolio")
    plt.legend(loc="upper left")
    plt.ylabel("Portfolio value")
    plt.xlabel("Timesteps / days")
    plt.show()
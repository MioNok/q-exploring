import numpy as np
import models
import random
import argparse
import functions as func

from keras.models import load_model
from keras.optimizers import Adam


#Model to test on:
LOAD_MODEL = "models/128x64.20c_RewSha-0.2_D-0.3___477.00max__-60.91avg_-478.00min__1579573684ep_19700mod_MLP.model" # Load existing model?. Insert path.
MODEL_NAME="Test"
MODEL_TYPE="MLP"
#Input Constants.
AGGREGATE_STATS_EVERY = 1
STOCK_DATA_FILE = "data/Dow2010-2019data.csv" #Filename for the data used for training
TICKER_FILE = "data/dowtickers.txt" #Filename for the symbols/tickers

#Reduce these to reduce the data trained on.
LIMIT_DATA = 500 # there is about 2500 datapoints(days) for each stock.
LIMIT_STOCKS = 1 #There is 10 years data for 30 stocks. Choose on how many you want to train.
NUMBER_OF_CANDLES = 20

###
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000

settings = {"Model_name": MODEL_NAME,
            "Stock_data_file": STOCK_DATA_FILE,
            "Ticker_file": TICKER_FILE,
            "Load_model": LOAD_MODEL,
            "Number_of_candles":NUMBER_OF_CANDLES,
            "Replay_memory_size": REPLAY_MEMORY_SIZE,
            "Aggregate_stats_every":AGGREGATE_STATS_EVERY,
            "Limit_data": LIMIT_DATA,
            "Limit_stocks":LIMIT_STOCKS,
            "Model_type":MODEL_TYPE}


def main(stock, aphkey):

    if stock != None:
        settings["Stock_data_file"] = "testdata/testdata.csv"
        settings["Ticker_file"] = "testdata/testticker.txt"
        func.fetchstockdata(aphkey,True,stock,None)



    #Make stock env.
    env = models.StockEnv(settings, True)
    agent = models.DQNAgent(env,settings)

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)

    # For stats
    ep_rewards = [0]

    episode_reward = 0
    step = 1
    current_state = env.reset()
    
    done = False

    while not done:
        action = agent.get_action(current_state)
        
        #Get simplestrat action
        simplestrat_action = func.simplestrat(current_state,settings)
        
        new_state, reward , done = env.step(action, 1, simplestrat_action)
        episode_reward += reward
       
        current_state = new_state
        step+=1

    print("Exiting.")

#main()

#TODO: Search for any stock and predict on that.
def parseargs():
    parser = argparse.ArgumentParser()


    #Arguments
    #Must haves
    parser = argparse.ArgumentParser()
    #None
 
    #Optional
    parser.add_argument("-s","--stock", help= "Stock ticker", type = str)
    parser.add_argument("-aph","--aphkey", help= "alphavantage apikey", type = str)
    
    args = parser.parse_args()
    aphkey = args.aphkey
    stock = args.stock

    return  stock, aphkey
    

if __name__ == "__main__":
    stock, aphkey = parseargs()
    main(stock, aphkey)
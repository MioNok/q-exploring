import numpy as np
import models
import random
import argparse
import functions as func

from keras.models import load_model
from keras.optimizers import Adam


#Model to test on:
LOAD_MODEL = "models/256x256.20c_RewSha-0.5_DO_0.4D-0.99____45.10max___11.27avg__-25.95min__1584554332ep_9700mod_MLP.model" # Load existing model?. Insert path.
MODEL_NAME="Test_256x256.20c_RewSha-0.5_DO_0.4D-0.99_test_serie_sp500_1"
MODEL_TYPE="MLP"
#Input Constants.
AGGREGATE_STATS_EVERY = 1
STOCK_DATA_FILE = "data/SP500-100-2018-2019_1_AdjustedData.csv" #Filename for the data used for training
TICKER_FILE = "data/SP500-100tickers.txt" #Filename for the symbols/tickers

#Reduce these to reduce the data trained on.
LIMIT_DATA = 2000
OFFSET_DATA = 0
LIMIT_STOCKS = 130
NUMBER_OF_CANDLES = 20
SKIP_STOCK = 0

###
REPLAY_MEMORY_SIZE = 2500
MIN_REPLAY_MEMORY_SIZE = 1000

#Only use stocks that have the wished amount of data and no less?
FULLTEST = False

settings = {"Model_name": MODEL_NAME,
            "Stock_data_file": STOCK_DATA_FILE,
            "Ticker_file": TICKER_FILE,
            "Load_model": LOAD_MODEL,
            "Number_of_candles":NUMBER_OF_CANDLES,
            "Replay_memory_size": REPLAY_MEMORY_SIZE,
            "Aggregate_stats_every":AGGREGATE_STATS_EVERY,
            "Limit_data": LIMIT_DATA,
            "Limit_stocks":LIMIT_STOCKS,
            "Model_type":MODEL_TYPE,
            "Skip_stock":SKIP_STOCK,
            "Offset_data":OFFSET_DATA,
            "Fulltest":FULLTEST}


def main(stock, aphkey):

    if stock != None:
        settings["Stock_data_file"] = "testdata/testdata.csv"
        settings["Ticker_file"] = "testdata/testticker.txt"
        func.fetchstockdata(aphkey,True,stock,None)


    #Make stock env.
    env = models.StockEnv(settings, preview =True)
    agent = models.DQNAgent(env,settings)

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)

    # For stats
    ep_rewards = [0]

    #Running for 2x the amount of stocks + 1 to make sure the episode stops at 200.
    for episode in range(LIMIT_STOCKS+1):
            
        done = False
        episode_reward = 0
        step = 0
        current_state = env.reset(rand = False)

        while not done:
            action = agent.get_action(current_state)
            
            #Get simplestrat action
            simplestrat_action = func.simplestrat(current_state,settings)
            
            new_state, reward , done = env.step(action, episode, simplestrat_action)
            episode_reward += reward
        
            current_state = new_state
            step+=1
        print("Ep done ", episode)

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
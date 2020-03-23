import numpy as np
import models
import random
import argparse
import functions as func

from keras.models import load_model
from keras.optimizers import Adam


###Portfolio testing###

#Model to test on:
LOAD_MODEL = "models/256x256.20c_RewSha-0.5_DO_0.4D-0.99____45.10max___11.27avg__-25.95min__1584554332ep_9700mod_MLP.model" # Load existing model?. Insert path.
MODEL_NAME="Test_256x256.20c_RewSha-0.5_DO_0.4D-0.99_test_serie_sp500_1"
MODEL_TYPE="MLP"
#Input Constants.
AGGREGATE_STATS_EVERY = 1
STOCK_DATA_FILE = "data/SP500-100_All_Data_1.csv" #Filename for the data used for training
TICKER_FILE = "data/SP500-100tickers.txt" #Filename for the symbols/tickers

#Reduce these to reduce the data trained on.
LIMIT_DATA = 5000
OFFSET_DATA = 0
LIMIT_STOCKS = 50
NUMBER_OF_CANDLES = 20
SKIP_STOCK = 0

###
REPLAY_MEMORY_SIZE = 2500
MIN_REPLAY_MEMORY_SIZE = 1000

#Only use stocks that have the wished amount of data and no less?
FULLTEST = True

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


def main():


    #Make stock env.
    env = models.StockEnv(settings, preview =False)
    agent = models.DQNAgent(env,settings)

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)


    step = 0
    stock_counter = 0

    #Loop
    for episode in range((LIMIT_DATA-NUMBER_OF_CANDLES)*env.preprosNumStocks):
            
        current_state, step = env.portReset(episode,env.preprosNumStocks)

        action = agent.get_action(current_state)
        
        env.portTestStep(action, episode, step)
        
        stock_counter +=1

        if stock_counter > env.preprosNumStocks:
            step += 1
            stock_counter = 0
            print("Step done ", step, "/", LIMIT_DATA-NUMBER_OF_CANDLES)

        

    print("Plotting.")
    func.plotporttestdata(MODEL_NAME,MODEL_TYPE)





if __name__ == "__main__":
    main()
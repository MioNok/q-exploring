import numpy as np
import models
import random

from keras.models import load_model


#Model to test on:
LOAD_MODEL = "models/256x256x32.50c__-580.88max_-2751.33avg_-5372.70min__1578261570ep_25.model" # Load existing model?. Insert path.
MODEL_NAME="Test"
#Input Constants.
AGGREGATE_STATS_EVERY = 1
STOCK_DATA_FILE = "Dow2010-2019data.csv" #Filename for the data used for training
TICKER_FILE = "dowtickers.txt" #Filename for the symbols/tickers

#Reduce these to reduce the data trained on.
LIMIT_DATA = 2500 # there is about 2500 datapoints(days) for each stock.
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
            "Limit_stocks":LIMIT_STOCKS}


def main():

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
        new_state, reward , done = env.step(action, 1)
        episode_reward += reward
       
        current_state = new_state
        step+=1

    print("Exiting.")

main()

#TODO: Search for any stock and predict on that.
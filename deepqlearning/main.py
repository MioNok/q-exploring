import functions as func
import pandas as pd
import argparse
import models 
import os
from tqdm import tqdm
from collections import deque
import time
import numpy as np
import random
import tensorflow as tf
import test

#Input Constants.
START_EPSILON = 1
EPSILON_DECAY = 0.99965 #Old 0.99965
MIN_EPSILON = 0.05
EPISODES = 30000
AGGREGATE_STATS_EVERY = 100
STOCK_DATA_FILE = "data/SP100_2015-2019data.csv" #Filename for the data used for training
TICKER_FILE = "data/SP500-100tickers.txt" #Filename for the symbols/tickers

LOAD_MODEL = None #"models/128x128.20c_RewSha-0.5_DO_0.4_0.15D-0.99____99.99max___-6.91avg__-46.57min__1584811669ep_2000mod_MLP.model" # Load existing model?. Insert path.
REPLAY_MEMORY_SIZE = 5000 #old 2500
MIN_REPLAY_MEMORY_SIZE = 1000

MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

#How many candles should the prediction be made on?
NUMBER_OF_CANDLES = 20

MODEL_NAME="64x64."+str(NUMBER_OF_CANDLES)+"c_RewSha-0.5_DO_0.4_0.15D-"+str(DISCOUNT)
MODEL_TYPE ="MLP" #Currently MLP(Fully connected) or LSTM or CNN"

#Reduce these to reduce the data trained on.
LIMIT_DATA = 584 # Days of data for training
OFFSET_DATA = 170 # This gives us 2017-01-03 to 2019-04-30 (Limit 584, offset 170)
LIMIT_STOCKS = 98 # Number of stocks to train on (-DD, -DOW)
SKIP_STOCK = 0

#Only use stocks that have the wished amount of data and no less?
FULLTEST = False


settings = {"Model_name": MODEL_NAME,
            "Stock_data_file": STOCK_DATA_FILE,
            "Ticker_file": TICKER_FILE,
            "Load_model": LOAD_MODEL,
            "Replay_memory_size": REPLAY_MEMORY_SIZE,
            "Min_replay_memory_size": MIN_REPLAY_MEMORY_SIZE,
            "Minibatch_size": MINIBATCH_SIZE,
            "Discount": DISCOUNT,
            "Update_target_every":UPDATE_TARGET_EVERY,
            "Number_of_candles":NUMBER_OF_CANDLES,
            "Aggregate_stats_every":AGGREGATE_STATS_EVERY,
            "Limit_data": LIMIT_DATA,
            "Limit_stocks":LIMIT_STOCKS,
            "Model_type":MODEL_TYPE,
            "Skip_stock":SKIP_STOCK,
            "Offset_data":OFFSET_DATA,
            "Fulltest":FULLTEST}


#Run this
def main(aphkey,data,preview):

    #If data is flagged, fetch it, else use the file attached. Currenttly not really in use.
    if data:
        func.fetchstockdata(aphkey,False,None,settings["Ticker_file"])
        exit()

    env = models.StockEnv(settings, preview)
    agent = models.DQNAgent(env,settings)

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)

    # For stats
    ep_rewards = [0]

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')


    epsilon = START_EPSILON
    for episode in tqdm(range(1, EPISODES+1), ascii = True, unit= "episodes"):
        
        agent.tensorboard.step = episode
    
        episode_reward = 0
        step = 1
        current_state = env.reset(rand= True) #>We want to randomise which stock comes.
    
        done = False
    
        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))

            else:
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)

            #Get simplestrat action, currently not in use.
            #simplestrat_action = func.simplestrat(current_state,settings)

            new_state, reward , done = env.step(action, episode, 1) # 1 is the simple strat action placeholder
    
            episode_reward += reward
        
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done,step)
    
            current_state = new_state
            step+=1
    
    
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            #Save model. Dont care about the early models.
            #if episode > 10000 or LOAD_MODEL != None:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}ep_{episode}epsi_{epsilon}mod_{settings["Model_type"]}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
    

def parseargs():
    parser = argparse.ArgumentParser()


    #Arguments
    #Must haves
    parser = argparse.ArgumentParser()
    #None
 
    #Optional 
    parser.add_argument("-d","--data",help="fetch data from api", action='store_true')
    parser.add_argument("-p","--preview",help="preview graphs", action='store_true')
    parser.add_argument("-g","--gpu",help="Enable gpu settings", action='store_true')
    parser.add_argument("-aph","--aphkey", help= "alphavantage apikey", type = str)
    

    
    args = parser.parse_args()
    aphkey = args.aphkey
    gpu = args.gpu
    preview = args.preview
    data = args.data


    ##### TF gpu settings.
    if gpu:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    return  aphkey, data, preview 
    

if __name__ == "__main__":
    aphkey, data, preview = parseargs()
    main(aphkey, data, preview)
    

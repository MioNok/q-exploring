#import functions as func
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

EPSILON_DECAY = 0.999
MIN_EPSILON = 0.05
EPISODES = 2500
AGGREGATE_STATS_EVERY = 30
MODEL_NAME="128x64x32."


##### TF gpu settings.
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)



#Run this
def main(aphkey,data):

    #TODO fix later.
    #If data is flagged, fetch it, else use the file attached.
    #if data:
    #    stockdata = func.fetchdowstockdata(aphkey)
    #else:
    #    stockdata = pd.read_csv("Dow2010-2019data.csv")

    env = models.StockEnv()
    agent = models.DQNAgent(env)

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)

    # For stats
    ep_rewards = [0]

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')


    epsilon = 1
    for episode in tqdm(range(1, EPISODES+1), ascii = True, unit= "episodes"):
        
        agent.tensorboard.step = episode
    
        episode_reward = 0
        step = 1
        current_state = env.reset()
    
        done = False
    
        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))

            else:
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)

            new_state, reward , done = env.step(action)
    
            episode_reward += reward
        
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done,step)

            if done:
                print(episode_reward)
    
            current_state = new_state
            step+=1
    
    
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            #if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
    

def parseargs():
    parser = argparse.ArgumentParser()


    #Arguments
    #Must haves
    parser = argparse.ArgumentParser()
    parser.add_argument("-aph","--aphkey", help= "alphavantage apikey", required = True, type = str)
 
    #Optional 
    parser.add_argument("-d","--data",help="fetch data from api", action='store_true')

    
    args = parser.parse_args()
    aphkey = args.aphkey
    data = args.data

    return  aphkey, data
    

if __name__ == "__main__":
    aphkey, data = parseargs()
    main(aphkey, data)
    

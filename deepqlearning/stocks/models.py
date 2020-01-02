

#import keras.backend.tensorflow_backend as backend
from tensorflow.python.keras import backend 
from tensorflow.keras import backend

import tensorflow as tf

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import TensorBoard
from keras.optimizers import Adam


from collections import deque
import time
import numpy as np
import pandasql as ps
import pandas as pd
import random

# TODO Move as a flags?
LOAD_MODEL = None
#LOAD_MODEL = "models/128x64x32__36148.23max_23448.90avg_14896.44min__1577897492.model"
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MODEL_NAME="128x64x32.x"

MINIBATCH_SIZE = 64
DISCOUNT = 0.9

UPDATE_TARGET_EVERY = 5


class Portfolio:
    def __init__(self):
        self.portfolio = {"ticker":["AAPL"],
                          "share": [0],
                          "currentstockvalue": [1],
                          "unusedBP": 10000}
        
    def action(self,action,current_step,stockdata,NUM_CANDLES):
        
        if action == 0: #sell
            current_observation = stockdata.iloc[current_step:current_step+NUM_CANDLES,]
            #print(current_observation)
            #print(current_step)
            
            #Close value of the stock for the current observation
            close_value = current_observation.close.iloc[-1]
            
            #check if we have shares.
            if self.portfolio["share"][0] > 0:
                #sell everything
                
                #Amount of shares held * price sold at, set current shares held to 0
                amount_to_be_credited = self.portfolio["share"][0]*close_value
                self.portfolio["share"][0] = 0
                
                #Add the sum to unusedBP
                self.portfolio["unusedBP"] += amount_to_be_credited
                
                #Update current stock value
                self.portfolio["currentstockvalue"] = close_value
            #else:
                #print("No shares","numshares",self.portfolio["share"][0])

        elif action == 1: #hold
            pass #Do nothing.
                
        elif action == 2: #buy
            current_observation = stockdata.iloc[current_step:current_step+NUM_CANDLES,]
            #print(current_observation)
            #print(current_step)
            
            #Close value of the stock for the current observation
            close_value = current_observation.close.iloc[-1]
            
            #check if we have shares.
            if self.portfolio["share"][0] <= 0:
                #Buy as much as possible everything
                
                #Amount of shares bought * price bought at, set current shares held to 0
                
                amount_of_shares_bought = int(self.portfolio["unusedBP"]/close_value)
                self.portfolio["share"][0] = amount_of_shares_bought
                
                #Substartct the sum to unusedBP
                self.portfolio["unusedBP"] -= amount_of_shares_bought*close_value
                
                #Update current stock value
                self.portfolio["currentstockvalue"] = close_value
            #else: 
                #print("We already have shares","numshares",self.portfolio["share"][0])
                
        
        
        

class StockEnv:
    def __init__(self):
        self.current_portfolio = dict()

        #The data
        stoset = pd.read_csv("Dow2010-2019data.csv")
        query = """SELECT timestamp, open, high, low, close, volume FROM stoset WHERE ticker = 'MCD' LIMIT 2500"""
        self.stoset = ps.sqldf(query).iloc[::-1]
        stoset_notimestamp = self.stoset.iloc[:,1:6]
    
        #Normalizing using minmax normalzation
        self.normalized_stoset =(stoset_notimestamp-stoset_notimestamp.min())/(stoset_notimestamp.max()-stoset_notimestamp.min())
    
        self.NUM_CANDLES = 20 
        self.ACTION_SPACE_SIZE = 3 #sell = 0, hold = 1, buy = 2
        self.OBSEREVATION_SPACE_VALUES = (self.NUM_CANDLES, 5) # 20 candles, observations for each candle, OHLC + volume.
        self.MAX_STEPS = 2500 #Currentlty the size of our dataset for each stock
        self.current_step = 0 # will update as we go
    
    
    
    def reset(self):
        #reset and return first observation
        self.current_portfolio = Portfolio()
        
        #first observation
        self.current_step = 0
        observation = self.get_data()
        
        return observation
    
    def step(self, action):
        self.current_step +=1
        self.current_portfolio.action(action, self.current_step, self.stoset, self.NUM_CANDLES)
        
        next_observation = self.get_data()
        
        #Check the reward. The value of the portfolio is equal to the size of the reward.
        reward = 0
        if self.current_step == self.MAX_STEPS-self.NUM_CANDLES:
            reward = self.current_portfolio.portfolio["share"][0] * self.current_portfolio.portfolio["currentstockvalue"] + self.current_portfolio.portfolio["unusedBP"]
        
        #Not done untill we reach the finnish
        done = False
        if self.current_step == self.MAX_STEPS-self.NUM_CANDLES:
            done = True
        
        #print(self.current_portfolio.portfolio)
        #time.sleep(2)
        return next_observation, reward, done
    
    def get_data(self): 
        current_observation = np.array(self.normalized_stoset.iloc[self.current_step:self.current_step+self.NUM_CANDLES,])
        
        return current_observation
    
    def get_current_portfolio(self):
        
        return self.current_portfolio


#From Sentdex
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)





#From Sentdex
# Agent class
class DQNAgent:
    def __init__(self,env):
        
        #main model gets trained
        self.env = env
        self.model = self.create_model()
        
        #target model use this for predict
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        self.target_update_counter = 0
        

    def create_model(self):

        if  LOAD_MODEL is not None:
            print("Loading", LOAD_MODEL)
            model = load_model(LOAD_MODEL)
            print("Loaded model", LOAD_MODEL)
        else:
            model = Sequential()
            model.add(Dense(128, input_shape = self.env.OBSEREVATION_SPACE_VALUES))
            model.add(Activation("relu"))
            model.add(Dropout(0.2))
            
            model.add(Dense(64))
            model.add(Activation("relu"))
            model.add(Dropout(0.2))
            
            model.add(Dense(32))
            model.add(Dense(self.env.ACTION_SPACE_SIZE, activation = "linear"))
            model.compile(loss = "mse", optimizer = Adam(lr=0.001), metrics=["accuracy"])
        return model
    
    def update_replay_memory(self,transition):
        self.replay_memory.append(transition)
        
        
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    
    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)
                
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)
        
        x= []
        y= []
        
        for index, (current_state, action ,reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            
            current_qs = current_qs_list[index]
            if action > 3:
                print("Action > 3",action)
            try:
                current_qs[action] = new_q
            except:
                print("Index Error" )
                print(current_qs,"current_qs")
                print(new_q,"new_q")
                print("action", action)
                exit()
            
            x.append(current_state)
            y.append(current_qs)
        
        self.model.fit(np.array(x)/255, np.array(y), batch_size = MINIBATCH_SIZE, verbose = 0, shuffle = False, callbacks = [self.tensorboard] if terminal_state else None)

         #updating to determin if we weant to update target model
        if terminal_state:
            self.target_update_counter +=1
            
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
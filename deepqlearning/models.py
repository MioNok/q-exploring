
#ML stuff
from tensorflow.python.keras import backend 
from tensorflow.keras import backend
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

#Other libraries
from collections import deque
import time
import numpy as np
import pandasql as ps
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from datetime import datetime

#Program files
import functions as func


class Portfolio:
    def __init__(self):
        self.portfolio = {"ticker":["TBA"], #Ticker is not used but here.
                          "share": [0],
                          "currentstockvalue": [1],
                          "unusedBP": [10000],
                          "porthistory": [[10000]]}
        
    def action(self,action,current_step,stockdata,NUM_CANDLES,current_stock):
        
        if action == 0: #sell
            current_observation = stockdata.iloc[current_step:current_step+NUM_CANDLES,]
            
            #Close value of the stock for the current observation
            close_value = current_observation.close.iloc[-1]
            
            #check if we have shares.
            if self.portfolio["share"][current_stock] > 0:
                #sell everything
                
                #Amount of shares held * price sold at, set current shares held to 0
                amount_to_be_credited = self.portfolio["share"][current_stock]*close_value
                self.portfolio["share"][current_stock] = 0
                
                #Add the sum to unusedBP
                self.portfolio["unusedBP"][current_stock] += amount_to_be_credited
                
            #Update current stock value
            self.portfolio["currentstockvalue"][current_stock] = close_value

            #Update portfolio history
            self.portfolio["porthistory"][current_stock].append(self.portfolio["unusedBP"][current_stock]+ self.portfolio["share"][current_stock] * self.portfolio["currentstockvalue"][current_stock] )


        elif action == 1: #hold

            current_observation = stockdata.iloc[current_step:current_step+NUM_CANDLES,]

            #Close value of the stock for the current observation
            close_value = current_observation.close.iloc[-1] 

            #Update portfolio history
            self.portfolio["porthistory"][current_stock].append(self.portfolio["unusedBP"][current_stock]+ self.portfolio["share"][current_stock] * self.portfolio["currentstockvalue"][current_stock] )
           
            #Update current stock value
            self.portfolio["currentstockvalue"][current_stock] = close_value
             #Nothing else
            
                
        elif action == 2: #buy
            current_observation = stockdata.iloc[current_step:current_step+NUM_CANDLES,]
            
            #Close value of the stock for the current observation
            close_value = current_observation.close.iloc[-1]
            
            #check if we have shares.
            if self.portfolio["share"][current_stock] <= 0:
                #Buy as much as possible everything
                
                #Amount of shares bought * price bought at, set current shares held to 0
                
                amount_of_shares_bought = int(self.portfolio["unusedBP"][current_stock]/close_value)
                self.portfolio["share"][current_stock] = amount_of_shares_bought
                
                #Substartct the sum to unusedBP
                self.portfolio["unusedBP"][current_stock] -= amount_of_shares_bought*close_value
                
            #Update current stock value
            self.portfolio["currentstockvalue"][current_stock] = close_value

            #Update portfolio history
            self.portfolio["porthistory"][current_stock].append(self.portfolio["unusedBP"][current_stock]+ self.portfolio["share"][current_stock] * self.portfolio["currentstockvalue"][current_stock])


                
    def new_stock(self):
        self.portfolio["ticker"].append("TBA")
        self.portfolio["share"].append(0)
        self.portfolio["currentstockvalue"].append(0)
        self.portfolio["unusedBP"].append(10000)
        self.portfolio["porthistory"].append([10000])     
        
        

class StockEnv:
    def __init__(self,settings, preview):

        self.current_portfolio = dict()
        self.buy_n_hold_portfolio = dict()
        self.settings = settings

        self.amount_of_stocks = settings["Limit_stocks"]
        self.preview = preview
        self.stoset , self.normalized_stoset = func.preprocessdata(settings["Stock_data_file"],
                                                                   settings["Ticker_file"],
                                                                   settings["Limit_data"], 
                                                                   settings["Limit_stocks"])
        self.NUM_CANDLES = settings["Number_of_candles"] 
        self.ACTION_SPACE_SIZE = 3 #sell = 0, hold = 1, buy = 2
        self.OBSEREVATION_SPACE_VALUES = (self.NUM_CANDLES, 5) # 20 candles, observations for each candle, OHLC + volume.
        self.MAX_STEPS = 2500 #Currentlty the size of our dataset for each stock
        self.current_step = 0 # will update as we go

        self.current_stock = 0 # Training data holds 30 stocks
        self.stock_size = 2500 # default is 2500, but for some stocks there is not that much data available
        
    
    
    def reset(self):
        #reset and return first observation
        self.current_portfolio = Portfolio()
        self.buy_n_hold_portfolio = Portfolio()
        
        #first observation
        self.current_step = 0
        observation = self.get_data()
        
        return observation
    
    
    def step(self, action, episode):
        self.current_step +=1

        #Update the stock size
        self.stock_size = self.stoset[self.current_stock].shape[0]

        #update current_portfolio
        self.current_portfolio.action(action, self.current_step, self.stoset[self.current_stock], self.NUM_CANDLES, self.current_stock)

        #The Benchmark
        #update buy and hold portfolio. It always buys an then holds untill the next stock.
        self.buy_n_hold_portfolio.action(2, self.current_step, self.stoset[self.current_stock], self.NUM_CANDLES, self.current_stock)

        next_observation = self.get_data()
        
        #Check the reward. The value of the portfolio is equal to the size of the reward.
        reward = 0
        if self.current_step == self.stock_size-self.NUM_CANDLES-1:
            current_port_sum = self.current_portfolio.portfolio["share"][self.current_stock] * self.current_portfolio.portfolio["currentstockvalue"][self.current_stock] + self.current_portfolio.portfolio["unusedBP"][self.current_stock]
            benchmark_port_sum = self.buy_n_hold_portfolio.portfolio["share"][self.current_stock] * self.buy_n_hold_portfolio.portfolio["currentstockvalue"][self.current_stock] + self.buy_n_hold_portfolio.portfolio["unusedBP"][self.current_stock]
            

            #The final reward is the difference in the gain between the ML algo and the benchmark
            reward = current_port_sum - benchmark_port_sum

            #If preview is set to true, save graph of the performace.
            # Dont save it every round, only if the episode is divisible by Aggregate_stats_every
            if self.preview and not episode % self.settings["Aggregate_stats_every"]:
                #Port values.
                current_port_history = np.array(self.current_portfolio.portfolio["porthistory"][self.current_stock])
                benchmark_port_history = np.array(self.buy_n_hold_portfolio.portfolio["porthistory"][self.current_stock])
                
                #Dates to graphs, ignore the first 50 since they are also ignored in the porthistory.
                timeseries = np.array(self.stoset[self.current_stock].timestamp[self.NUM_CANDLES:])
                dates = np.array([datetime.strptime(day, '%Y-%m-%d') for day in timeseries])

                # Create graphs folder
                if not os.path.isdir('graphs'):
                    os.makedirs('graphs')


                #Plot graphs and show them
                plt.figure(figsize=(20,10))
                plt.plot(dates,current_port_history, label = "Reinforced portfolio")
                plt.plot(dates,benchmark_port_history, label = "Benchmark portfolio")
                plt.legend(loc="upper left")
                
                #Save graph to folder
                
                plt.savefig("graphs/Fig_stock"+str(self.current_stock)+"_ep"+str(episode)+".png")
                #Uncomment to show graphs as they come instead.
                #plt.show(block=False)
                #plt.pause(5)
                plt.close()

            #Stock is finnished, give reward, reset steps and move to next stock
            #Reset the steps and add to stock unless we are at we are at the last stock. Env.reset will reset it
            #And this last step is needed for the next if statement for us to know if we have reached the end.
            if self.current_stock != self.amount_of_stocks-1:
                self.current_step = 0
                self.current_stock += 1
  
        

            #Set up the portfolio to accept a new stock
            self.current_portfolio.new_stock()
            self.buy_n_hold_portfolio.new_stock()


        #Not done untill we reach the finnish
        done = False
        if self.current_step == self.stock_size-self.NUM_CANDLES-1 and self.current_stock == self.amount_of_stocks-1:
            done = True

            #Done with the loop, set current stock to 0 for the next one.
            self.current_stock = 0
        
        return next_observation, reward, done
    
    def get_data(self):
        #Get the data 
        current_observation = self.stoset[self.current_stock].iloc[self.current_step:self.current_step+self.NUM_CANDLES,] 
    
        #Remove the timestamp
        current_observation_no_timestamp = current_observation.iloc[:,1:6]

        #Normalize the observation
        normalized_observation =(current_observation_no_timestamp-current_observation_no_timestamp.min())/(current_observation_no_timestamp.max()-current_observation_no_timestamp.min())

        return np.array(normalized_observation)

    
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
    def __init__(self,env, settings):
        
        #main model gets trained
        self.env = env
        self.settings = settings
        self.model = self.create_model()
        
        #target model use this for predict
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=settings["Replay_memory_size"] )
        
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(settings["Model_name"] , int(time.time())))
        self.target_update_counter = 0
        

    def create_model(self):

        if  self.settings["Load_model"] is not None: 
            print("Loading", self.settings["Load_model"])
            model = load_model(self.settings["Load_model"])
            print("Loaded model", self.settings["Load_model"])
        else:
            model = Sequential()
            model.add(Dense(256, input_shape = self.env.OBSEREVATION_SPACE_VALUES))
            model.add(Activation("relu"))
            model.add(Dropout(0.2))
            
            model.add(Dense(256))
            model.add(Activation("relu"))
            model.add(Dropout(0.2))

            model.add(Flatten())
            model.add(Dense(32))
            model.add(Dense(self.env.ACTION_SPACE_SIZE, activation = "linear"))
            model.compile(loss = "mse", optimizer = Adam(lr=0.001), metrics=["accuracy"])
        return model
    
    def update_replay_memory(self,transition):
        self.replay_memory.append(transition)
        
        
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
    
    def get_action(self, state):
        return self.model.predict_classes(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < self.settings["Min_replay_memory_size"]:
            return
        
        minibatch = random.sample(self.replay_memory, self.settings["Minibatch_size"])
        
        
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)
                
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)
        
        x= []
        y= []
        
        for index, (current_state, action ,reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.settings["Discount"] * max_future_q
            else:
                new_q = reward
            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            
            x.append(current_state)
            y.append(current_qs)
        
        self.model.fit(np.array(x), np.array(y), batch_size = self.settings["Minibatch_size"], verbose = 0, shuffle = False, callbacks = [self.tensorboard] if terminal_state else None)

         #updating to determin if we weant to update target model
        if terminal_state:
            self.target_update_counter +=1
            
        if self.target_update_counter > self.settings["Update_target_every"]:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
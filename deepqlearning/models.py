
#ML stuff
from tensorflow.python.keras import backend 
from tensorflow.keras import backend
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, CuDNNLSTM, BatchNormalization, MaxPooling2D, Conv2D
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
        self.portfolio = {"ticker":["TBA"], #Ticker is not used but here for the future.
                          "share": [0],
                          "currentstockvalue": [1],
                          "unusedBP": [10000],
                          "porthistory": [[10000]]}
    #Current stock can be quickly switched an number if we want to run multiple stock in the same episode.
    #Currently current stock is always 0 since we reset portfolios after each stock.
    def action(self,action,current_step,stockdata,NUM_CANDLES,current_stock=0):
       
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
        self.simplestrat_portfolio = dict()

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

        self.current_stock = -1 # Training data holds 100 stocks
        self.stock_size = 2500 # default is 2500, but for some stocks there is not that much data available
        self.last_reward = 0 # keep track if the change in the size of reward
        self.ep_reward = 0 # This episode reward is just for logging purposes.

        self.logdatafile = pd.DataFrame() #Dataframe to hold logs for the runs. Written to a file in logdata/


    def reset(self, rand):
        #reset and return first observation
        self.current_portfolio = Portfolio()
        self.buy_n_hold_portfolio = Portfolio()
        self.simplestrat_portfolio = Portfolio()
        
        #first observation
        self.current_step = 0
        observation = self.get_data()

        #Pick one stock at random if rand is true, othervise just go trough all one by one.
        if rand:
            self.current_stock = np.random.randint(0,self.settings["Limit_stocks"])
        else:
            self.current_stock += 1

        if self.current_stock == 100:
            self.current_stock = 0

        #Update the stock size and set ep reward to 0.
        self.stock_size = self.stoset[self.current_stock].shape[0]
        self.ep_reward = 0
        self.last_reward_pcr = 0

        return observation
    
    
    def step(self, action, episode,simplestrat_action):
        self.current_step +=1

        #self.stock_size = self.stoset[self.current_stock].shape[0]

        #update current_portfolio
        self.current_portfolio.action(action, self.current_step, self.stoset[self.current_stock], self.NUM_CANDLES, 0)

        #The Benchmark
        #update buy and hold portfolio. It always buys an then holds untill the next stock.
        self.buy_n_hold_portfolio.action(2, self.current_step, self.stoset[self.current_stock], self.NUM_CANDLES, 0)

        #Simple strat port
        simplestrat_action 
        self.simplestrat_portfolio.action(simplestrat_action, self.current_step, self.stoset[self.current_stock], self.NUM_CANDLES, 0)

        next_observation = self.get_data()
        
        #Reward shaping
        # Clarification - Reward == difference the value of the portfolios
        # Calculate after every step the percentage difference to the benchmark 
        # If the difference has increased the reward is 1, if it has decreased its -1
        current_port_sum = self.current_portfolio.portfolio["share"][0] * self.current_portfolio.portfolio["currentstockvalue"][0] + self.current_portfolio.portfolio["unusedBP"][0]
        benchmark_port_sum = self.buy_n_hold_portfolio.portfolio["share"][0] * self.buy_n_hold_portfolio.portfolio["currentstockvalue"][0] + self.buy_n_hold_portfolio.portfolio["unusedBP"][0]
        difference = current_port_sum - benchmark_port_sum
        # Percentage difference in the gain of benchmark to algo
        reward_pcr = (difference / benchmark_port_sum) *100 


        #Update last reward if it is the 1st step in an episode.
        if self.current_step == 1:
            self.last_reward_pcr = reward_pcr
            reward = 0
        
        #We are past the 1st step
        # If the move we made has the difference increasing, increase the reward
        elif reward_pcr > self.last_reward_pcr:
            reward = 1
        
        # If the move we made has the difference increasing, decrease the reward
        elif reward_pcr < self.last_reward_pcr:
            reward = -1

        #If the reward is the same as last reward, no increase or decrease.
        else:
            reward = 0
            
        #Update the last reward and episode reward
        self.last_reward_pcr = reward_pcr
        self.ep_reward += reward




        if self.current_step == self.stock_size-self.NUM_CANDLES-1:
            current_port_sum = self.current_portfolio.portfolio["share"][0] * self.current_portfolio.portfolio["currentstockvalue"][0] + self.current_portfolio.portfolio["unusedBP"][0]#Switch 0 to self.current_stock if training on multiple stock in one episode.
            benchmark_port_sum = self.buy_n_hold_portfolio.portfolio["share"][0] * self.buy_n_hold_portfolio.portfolio["currentstockvalue"][0] + self.buy_n_hold_portfolio.portfolio["unusedBP"][0]#Switch 0 to self.current_stock if training on multiple stock in one episode.
            simplestrat_port_sum = self.simplestrat_portfolio.portfolio["share"][0] * self.simplestrat_portfolio.portfolio["currentstockvalue"][0] + self.simplestrat_portfolio.portfolio["unusedBP"][0]


            #Logging some data to plot later
            self.logdatafile = func.appendlogdata(self.current_stock, episode, self.ep_reward, current_port_sum, benchmark_port_sum, reward_pcr, self.logdatafile)

            #Wtite the data to file every now and then.
            if episode % 25 == 0 or episode == 1:
                func.writelogdata(self.logdatafile, self.settings)

            #The final reward is the difference in the gain between the ML algo and the benchmark
            #Calculate percentage difference
            #difference = current_port_sum - benchmark_port_sum
            #reward = (difference / benchmark_port_sum) *100

            #If preview is set to true, save graph of the performace.
            # Dont save it every round, only if the episode is divisible by Aggregate_stats_every
            if self.preview and not episode % self.settings["Aggregate_stats_every"]:
                #Port values.
                current_port_history = np.array(self.current_portfolio.portfolio["porthistory"][0])#Switch 0 to self.current_stock if training on multiple stock in one episode.
                benchmark_port_history = np.array(self.buy_n_hold_portfolio.portfolio["porthistory"][0])#Switch 0 to self.current_stock if training on multiple stock in one episode.
                simplestrat_port_history = np.array(self.simplestrat_portfolio.portfolio["porthistory"][0])

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
                plt.plot(dates,simplestrat_port_history, label = "Simplestrat portfolio")
                plt.legend(loc="upper left")
                
                #Save graph to folder
                plt.savefig("graphs/Fig_stock"+str(self.current_stock)+"_ep"+str(episode)+"_type"+self.settings["Model_type"]+".png")
                #Uncomment to show graphs as they come instead.
                #plt.show(block=False)
                #plt.pause(5)
                plt.close()

            #Stock is finnished, give reward, reset steps and move to next stock
            #Reset the steps and add to stock unless we are at we are at the last stock. Env.reset will reset it
            #And this last step is needed for the next if statement for us to know if we have reached the end.
            #if self.current_stock != self.amount_of_stocks-1:
            #    self.current_step = 0
            #    self.current_stock += 1
  
        

            #Set up the portfolio to accept a new stock if we loop trough multiple stocks.
            #self.current_portfolio.new_stock()
            #self.buy_n_hold_portfolio.new_stock()


        #Not done untill we reach the finnish
        done = False
        if self.current_step == self.stock_size-self.NUM_CANDLES-1:
            done = True
        
        return next_observation, reward, done
    
    def get_data(self):
        #Get the data 
        current_observation = self.stoset[self.current_stock].iloc[self.current_step:self.current_step+self.NUM_CANDLES,] 
    
        #Remove the timestamp
        current_observation_no_timestamp = current_observation.iloc[:,1:6]

        #Normalize the observation
        normalized_observation =(current_observation_no_timestamp-current_observation_no_timestamp.min())/(current_observation_no_timestamp.max()-current_observation_no_timestamp.min())
        
        #Reshape data to fit CNN.
        if self.settings["Model_type"] =="CNN":
            #CNN..
            normalized_observation = np.asarray(normalized_observation)
            
            normalized_observation = normalized_observation.reshape(self.settings["Number_of_candles"],5,1)
        
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
        
        self.tensorboard = ModifiedTensorBoard(log_dir=f'logs/{settings["Model_name"]}__{int(time.time())}mod_{settings["Model_type"]}')
        self.target_update_counter = 0
        

    def create_model(self):

        if  self.settings["Load_model"] is not None: 
            print("Loading", self.settings["Load_model"])
            model = load_model(self.settings["Load_model"])
            print("Loaded model", self.settings["Load_model"])
        else:
            if self.settings["Model_type"] == "MLP":
                model = Sequential()
                model.add(Dense(64, input_shape = self.env.OBSEREVATION_SPACE_VALUES))
                model.add(Activation("relu"))
                model.add(Dropout(0.2))
            
                #model.add(Dense(128))
                #model.add(Activation("relu"))
                #model.add(Dropout(0.2))

                model.add(Flatten())
                model.add(Dense(32))
                model.add(Activation("relu"))
                model.add(Dense(self.env.ACTION_SPACE_SIZE, activation = "linear"))
                model.compile(loss = "mse", optimizer = Adam(lr=0.001), metrics=["accuracy"])

            elif self.settings["Model_type"] == "LSTM":
                model = Sequential()
                model.add(CuDNNLSTM(128, input_shape=self.env.OBSEREVATION_SPACE_VALUES, return_sequences=True))#CuDNNLSTM
                model.add(Dropout(0.2))
                model.add(BatchNormalization()) 

                model.add(CuDNNLSTM(128))#CuDNNLSTM, no activation
                model.add(Dropout(0.2))
                model.add(BatchNormalization())

                model.add(Dense(64, activation='relu'))
                model.add(Dropout(0.2))

                model.add(Dense(self.env.ACTION_SPACE_SIZE, activation='softmax'))
                model.compile(loss='sparse_categorical_crossentropy', optimizer= Adam(lr=0.001), metrics=['accuracy'])

            elif self.settings["Model_type"] == "CNN":
                model = Sequential()
                model.add(Conv2D(128,(2,2), input_shape=(self.settings["Number_of_candles"],5,1), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.2))

                #model.add(Conv2D(128,(2,2), activation='relu'))
                #model.add(BatchNormalization())
                #model.add(Dropout(0.2))

                model.add(Flatten())
                model.add(Dense(64, activation='relu'))
                model.add(Dropout(0.2))

                model.add(Dense(self.env.ACTION_SPACE_SIZE, activation='softmax'))
                model.compile(loss='mse', optimizer= Adam(lr=0.001), metrics=['accuracy'])

            else:
                print("Unknown model")
                exit()

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
        

        if self.settings["Model_type"] == "LSTM":
            y_new = []
            for row in y:
                y_new.append(np.argmax(row))
            y = np.array(y_new).reshape(self.settings["Minibatch_size"],1)
        
        self.model.fit(np.array(x), np.array(y), batch_size = self.settings["Minibatch_size"], verbose = 0, shuffle = False, callbacks = [self.tensorboard] if terminal_state else None)
        
        #updating to determin if we weant to update target model
        if terminal_state:
            self.target_update_counter +=1
            
        if self.target_update_counter > self.settings["Update_target_every"]:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
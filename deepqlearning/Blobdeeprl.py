from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

import keras.backend.tensorflow_backend as backend

import tensorflow as tf
from keras import backend
from tensorflow.keras import backend
import random

from tqdm import tqdm
import os
from PIL import Image
import cv2

from collections import deque
import time
import numpy as np

#LOAD_MODEL = "models/256x2____25.00max_-200.75avg_-483.00min__1573480765.model"
LOAD_MODEL = None

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
MODEL_NAME="256x2"
UPDATE_TARGET_EVERY = 5

MIN_REWARD = -200
#MEMORY_FRACTIon = 0.2

EPISODES = 20000

epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 100
SHOW_PREVIEW = False


#gpu options.
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class Blob():
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)
    
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def action(self,choice):
        
        if choice == 0:
            self.move(x=1, y=1)      
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)
        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)
        elif choice == 8:
            self.move(x=0, y=0)
    
    def move(self, x=False, y= False):
        if not x:
            self.x += np.random.randint(-1,2)            
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1,2)            
        else:
            self.y += y
        
        
        if self.x <0:
            self.x = 0
            
        elif self.x > self.size-1:
            self.x = self.size -1
        
        if self.y <0:
            self.y = 0
            
        elif self.y > self.size-1:
            self.y = self.size -1
            
class BlobEnv:
    SIZE = 20
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        self.enemy2 = Blob(self.SIZE)
        self.enemy3 = Blob(self.SIZE)
        self.enemy4 = Blob(self.SIZE)
        self.enemy5 = Blob(self.SIZE)
        self.enemy6 = Blob(self.SIZE)

        #Not sure why..
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)
        
        while self.enemy2 == self.player or self.enemy2 == self.food:
            self.enemy2 = Blob(self.SIZE)

        while self.enemy3 == self.player or self.enemy3 == self.food:
            self.enemy3 = Blob(self.SIZE)
        
        while self.enemy4 == self.player or self.enemy4 == self.food:
            self.enemy4 = Blob(self.SIZE)
        while self.enemy5 == self.player or self.enemy5 == self.food:
            self.enemy5 = Blob(self.SIZE)
        while self.enemy6 == self.player or self.enemy6 == self.food:
            self.enemy6 = Blob(self.SIZE)

        self.episode_step = 0

        #Else is only if we dont use images as input
        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy) +(self.player-self.enemy2) +(self.player-self.enemy3) +(self.player-self.enemy4) + (self.player-self.enemy5) +(self.player-self.enemy6)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #self.enemy.move()
        #self.food.move()
        ##############

        #Else is only if we dont use images as input
        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemy)+(self.player-self.enemy2) +(self.player-self.enemy3) +(self.player-self.enemy4) + (self.player-self.enemy5) +(self.player-self.enemy6)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.enemy2:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.enemy3:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.enemy4:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.enemy5:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.enemy6:
            reward = -self.ENEMY_PENALTY

        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(100)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]
        env[self.enemy2.x][self.enemy2.y] = self.d[self.ENEMY_N]
        env[self.enemy3.x][self.enemy3.y] = self.d[self.ENEMY_N]
        env[self.enemy4.x][self.enemy4.y] = self.d[self.ENEMY_N]
        env[self.enemy5.x][self.enemy4.y] = self.d[self.ENEMY_N]
        env[self.enemy6.x][self.enemy4.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
#tf.set_random_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


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

# Agent class
class DQNAgent:
    def __init__(self):
        
        #main model gets trained
        self.model = self.create_model()
        
        #target model use this for predict
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        self.target_update_counter = 0
        
        
    def create_model(self):

        if LOAD_MODEL is not None:
            print("Loading", LOAD_MODEL)
            model = load_model(LOAD_MODEL)
            print("Loaded model", LOAD_MODEL)
        else:
            model  = Sequential()
            model.add(Conv2D(256,(3,3), input_shape = env.OBSERVATION_SPACE_VALUES))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(2,2))
            model.add(Dropout(0.2))
            
            model.add(Conv2D(256,(3,3)))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(2,2))
            model.add(Dropout(0.2))
            
            model.add(Flatten())
            model.add(Dense(64))
            model.add(Dense(env.ACTION_SPACE_SIZE, activation = "linear"))
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
            current_qs[action] = new_q
            
            x.append(current_state)
            y.append(current_qs)
        
        self.model.fit(np.array(x)/255, np.array(y), batch_size = MINIBATCH_SIZE, verbose = 0, shuffle = False, callbacks = [self.tensorboard] if terminal_state else None)
        
        
        #updating to determin if we weant to update target model
        if terminal_state:
            self.target_update_counter +=1
            
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
   

         

env = BlobEnv()
agent = DQNAgent()


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
    
        if SHOW_PREVIEW and not episode % 10:
            env.render()
        
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

        # Save model, but only when min reward is greater or equal a set value
        #if min_reward >= MIN_REWARD:
        agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
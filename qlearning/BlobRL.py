import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")


SIZE = 8
EPISODES = 300000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25

epsilon = 1
EPS_DECAY = 0
SHOW_EVERY = 1

start_q_table = "qtable-1573486800.pickle" #None or filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255) }


class Blob():
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
    
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
    
    def action(self,choise):
        
        if choise == 0:
            self.move(x =1, y=0)
        elif choise == 1:
            self.move(x=-1, y=0)
        elif choise == 2:
            self.move(x=0, y=1)
        elif choise == 3:
            self.move(x=0, y=-1)
    
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
            
        elif self.x > SIZE-1:
            self.x = SIZE -1
        
        if self.y <0:
            self.y = 0
            
        elif self.y > SIZE-1:
            self.y = SIZE -1
        
        
print("creating table")
if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        (print("Dingg"))
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    for x3 in range(-SIZE+1, SIZE):
                        for y3 in range(-SIZE+1, SIZE):
                            q_table[((x1,y1),(x2,y2),(x3,y3))] = [np.random.uniform(-5,0) for i in range(4)]
    #q_table = np.zeros([SIZE*SIZE*SIZE*SIZE*SIZE, 4]) #

else:
    with open(start_q_table,"rb") as f:
        q_table = pickle.load(f)

print("Table done/imported")
        
episode_rewards = []

for episode in range(EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()
    enemy2 = Blob()
    
    if episode % SHOW_EVERY == 0:
        print("On episode", episode, "epsilon:" , epsilon)
        print(SHOW_EVERY, "ep mean", np.mean(episode_rewards[-SHOW_EVERY:]))
        show= True
    
    else:
        show = False
        
        
    episode_reward = 0
    for i in range(200):
        obs = (player - food, player - enemy, player - enemy2)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
            #action = np.random.randint(0,4)
            
        else:
            action = np.random.randint(0,4)
            
        player.action(action)
        #print(action)
        
        enemy.move()
        enemy2.move()
        food.move()

        #If food is under enemy ->
        if enemy.x == food.x and enemy.y == food.y:
            food.move()
        if enemy2.x == food.x and enemy2.y == food.y:
            food.move()
        
        #Check penalties
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        
        if player.x == enemy2.x and player.y == enemy2.y:
            reward = -ENEMY_PENALTY
            
        elif player.x  == food.x and player.y ==food.y:
            reward = FOOD_REWARD
            
        else:
            reward = -MOVE_PENALTY
        
        new_obs = (player - food, player - enemy, player - enemy2)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]
        
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
       # elif reward == - ENEMY_PENALTY:
       #     new_q = -ENEMY_PENALTY
            
        else: 
            new_q = (1- LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            
        q_table[obs][action] = new_q
        
        
        if show:
            env = np.zeros((SIZE,SIZE,3), dtype=np.uint8)
            env[food.y][food.x] = d[FOOD_N]
            env[player.y][player.x] = d[PLAYER_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]
            env[enemy2.y][enemy2.x] = d[ENEMY_N]
            
            img = Image.fromarray(env, "RGB")
            img = img.resize((300,300))
            cv2.imshow("image",np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break               
                
            else:
                if cv2.waitKey(50) & 0xFF == ord("q"):
                    break
                
        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break
            
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/ SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(SHOW_EVERY)
plt.xlabel("Episode")
plt.show()

#with open("qtable-",str(int(time.time())),".pickle","wb") as f:
with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table,f)        
    
    

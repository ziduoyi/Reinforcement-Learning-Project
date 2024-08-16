import gymnasium as gym 
import pygame
from gymnasium import Env
from gymnasium import spaces
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import random, time
import os
#from stable_baseline3.stable_baselines3.ppo import ppo
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import torch as th
import torch.nn as nn
from torch.optim import Adam
from tensorflow import keras
import tensorflow as tf
from tensorflow import concat
import torch as th
import torch.nn as nn
from torch.optim import Adam
from collections import deque
import time
from segment_tree import MinSegmentTree, SumSegmentTree


class MowerEnv2():
    def __init__(self, info, render_mode = True): #info 0 = len, 1 = number of agents (up to 4)
        self.render_mode = render_mode
        self.len = info[0]
        self.agents = info[1]
        if render_mode == True:
            pygame.init()
            self.screen = pygame.display.set_mode((720, 720))
        # Actions we can up down left right
        # Set length
        self.direct = [[-1, 0], [0, 1], [1, 0], [0, -1]]

    def step(self, action):
        ret = []
        for j in range(self.agents):
            # Apply action (no walls for now)
            x = self.all[j][0] + self.direct[action[j]][0]
            y = self.all[j][1] + self.direct[action[j]][1]
            if self.isValid(x, y) == 0:
                x = self.all[j][0]
                y = self.all[j][1]
            if self.state[x][y][0] == 1:
                self.area_clear += 1
                reward = 5.0 * np.power(4, self.area_clear/self.max_area)
            else:
                reward = -0.5
                self.running_length -= 1
            self.state[x][y][0] = 0
            self.all[j][0] = x
            self.all[j][1] = y

            # update boundries
            for i in range(4):
                self.all[j][i+2] = self.isValid(x + self.direct[i][0], y + self.direct[i][1])
            temp = self.cheats(x, y)
            self.all[j][6]= self.all[j][7]= self.all[j][8]= self.all[j][9] = 0
            for p in temp:
                self.all[j][6+p] = 1
            # Check if  done
            if self.running_length < 0:
                done = True
            else:
                if self.finished(): #change back to self.finished()
                    if not done:
                        reward = self.max_area * 20
                    else:
                        reward = 0
                    done = True
                else:
                    done = False
            ret.append(reward)
        # Set placeholder for info
        info = {}

        # Return step information
        return (ret, done)

    def isValid(self, x, y):
        if x < 0 or x < 0 or x >= self.len or y < 0 or y >= self.len:
            return 0 #return False
        if self.state[x][y][1] == 1:
            return 0
        return 1 #return true

    def cheats(self, posx, posy):
        # Next action:
        # (feed the observation to your agent here)
        calc = np.zeros((16, 16))
        list = deque()
        calc[posx][posy] = 1
        list.append([posx, posy])
        move = []
        while True: #has multiple paths issues, but should be irrelevent
            size = len(list)
            for z in range(size):
                tuple = list.popleft()
                for i in range(4):
                    x = tuple[0] + self.direct[i][0]
                    y = tuple[1] + self.direct[i][1]
                    if x >= 0 and x < 16 and y >=0 and y < 16:
                        if self.state[x][y][1] == 0 and calc[x][y] == 0:
                            calc[x][y] = 1
                            choice = i
                            if len(tuple) == 3:
                                choice = tuple[2]
                            if self.state[x][y][0] == 0:
                                ret = [x, y, choice]
                                list.append(ret)
                            else:
                                move.append(choice)
                                break

            if len(move) > 0:
                break
        return move

    def finished(self):
        return self.area_clear==self.max_area

    def render(self):
        # Implement viz
        if self.render_mode == True:
            pygame.event.get()
            gap = 720//self.len
            for i in range(0, 720, gap):
                for j in range(0, 720, gap):
                    if self.state[i//gap][j//gap][0] == 0:
                        color = (255, 255, 255)
                    elif self.state[i//gap][j//gap][0] == 1:
                        color = (255, 255,0)
                    for k in range(self.agents):
                        if self.all[k][0] == i//gap and self.all[k][1] == j//gap:
                            color = (255, 0, 0)
                    if self.state[i//gap][j//gap][1] == 1:
                        color = (0, 0, 0)
                    pygame.draw.rect(self.screen, color, (j, i, gap, gap))
            pygame.display.update()


    def reset(self, seed=None, options=None):
        # Set starting state
        self.state = np.copy(combined_data)
        self.all = np.zeros((self.agents, 10), dtype = int)

        for i in range(self.agents):# Set position
            self.all[i][0] = corners[i][0]
            self.all[i][1] = corners[i][1]

            # Set border detect 0 = blocked, 1 = valid
            for j in range(4):
                self.all[i][j+2] = self.isValid(self.all[i][0] + self.direct[j][0], self.all[i][1] + self.direct[j][1])
            temp = self.cheats(self.all[i][0], self.all[i][1])
            self.all[i][6]= self.all[i][7]= self.all[i][8]= self.all[i][9] = 0
            for p in temp:
                self.all[i][6+p] = 1

        self.running_length = ((self.len) * (self.len) * (1 + self.agents)) // 2
        self.area_clear = 0
        self.max_area = np.sum(grass_data)
        return self.state, {}

    def get_state(self):
        return (self.state, self.all)

    def close(self):
        pass

grass_data = [[0,0,1,0,1,0,1,0,0,1,1,0,1,0,0,0],
[0,0,1,0,1,0,1,1,0,1,0,0,1,0,0,1],
[1,1,0,0,1,0,0,0,1,1,0,0,0,1,0,0],
[0,0,0,1,0,1,0,1,0,0,0,1,1,0,1,0],
[0,1,1,0,0,0,1,0,0,1,0,0,1,0,0,0],
[1,1,0,0,1,0,0,0,0,0,1,1,1,0,0,0],
[1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0],
[0,1,1,0,0,0,1,0,0,0,1,0,1,0,0,0],
[0,0,0,1,1,1,0,0,0,0,1,0,1,0,0,0],
[1,0,0,1,0,1,0,1,0,1,1,0,1,0,1,0],
[1,0,0,0,1,1,1,0,1,0,0,0,0,0,1,0],
[0,1,0,0,1,0,1,1,1,0,1,1,0,0,0,0],
[1,0,0,1,1,0,0,1,0,1,0,0,0,0,1,0],
[1,0,0,1,0,0,1,0,1,0,1,1,1,1,0,0],
[1,0,1,0,1,1,0,1,0,1,0,0,1,0,0,0],
[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
walls_data = [[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
[0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0],
[0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0],
[0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0],
[0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1],
[0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0],
[0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0],
[0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0],
[0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0,0,1,0,1,1,0,0],
[0,0,1,1,0,0,0,0,0,1,0,0,0,0,1,0],
[0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0],
[0,0,1,0,1,1,0,0,0,1,0,0,0,0,1,0],
[0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]]
corners = [[0, 0], [1, 1], [1, 0], [0, 1]]
walls_data = [] #delete later
for i in range(16):
    walls_data.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
corners = [[0, 0], [1, 1], [1, 0], [0, 1]]
combined_data = []
for i in range(16):
    combined_data.append([])
    for j in range(16):
        combined_data[i].append([grass_data[i][j], walls_data[i][j]])
combined_data = np.array(combined_data, dtype = int)

#@title
class model(keras.Model):
    def __init__(self):
        super().__init__()
        self.model1 = keras.Sequential([
            keras.layers.Input((16, 16, 2)), #change later
            keras.layers.ZeroPadding2D(padding = (1, 1), data_format="channels_last"),
            keras.layers.Conv2D(filters = 16,kernel_size = (3,3),strides = (1,1),
activation='relu',data_format="channels_last", kernel_regularizer='l2'),
            keras.layers.BatchNormalization(),
            keras.layers.ZeroPadding2D(padding = (1, 1), data_format="channels_last"),
            keras.layers.Conv2D(filters = 16,kernel_size = (3,3),strides = (1,1),
activation='relu',data_format="channels_last", kernel_regularizer='l2'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last"),
            keras.layers.ZeroPadding2D(padding = (1, 1), data_format="channels_last"),
            keras.layers.Conv2D(filters = 16,kernel_size = (3,3),strides = (1,1),
activation='relu',data_format="channels_last", kernel_regularizer='l2'),
            keras.layers.BatchNormalization(),
            keras.layers.ZeroPadding2D(padding = (1, 1), data_format="channels_last"),
            keras.layers.Conv2D(filters = 16,kernel_size = (3,3),strides = (1,1),
activation='relu',data_format="channels_last", kernel_regularizer='l2'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last"),
            keras.layers.Flatten(data_format="channels_last"),
            keras.layers.Dense(units=64, activation="relu", kernel_regularizer='l2')
        ])
        self.model2 = keras.Sequential([
            keras.layers.Input((10,)),
            keras.layers.Dense(units=64, activation = 'relu', kernel_regularizer='l2')
        ])
        self.model3 = keras.Sequential([
            keras.layers.Input((128,)),
            keras.layers.Dense(units=64, activation="relu", kernel_regularizer='l2'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(units=64, activation="relu", kernel_regularizer='l2'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(units=4, activation="linear", kernel_regularizer='l2')
        ])
        """
        #The layers to process our image
        self.Padding_1 = keras.layers.ZeroPadding2D(data_format="channels_first")
        self.Conv2D_1 = keras.layers.Conv2D(filters = 64,
                                            kernel_size = (3,3),
                                            strides = (1,1),
                                            activation='relu',
                                            data_format="channels_first")
        self.MaxPooling_1 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.Padding_2 = keras.layers.ZeroPadding2D(data_format="channels_first")
        self.Conv2D_2 = keras.layers.Conv2D(filters = 64,
                                            kernel_size = (3,3),
                                            strides = (1,1),
                                            activation='relu',
                                            data_format="channels_first") 
        self.MaxPooling_2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.Conv2D_3 = keras.layers.Conv2D(filters = 64,
                                            kernel_size = (2,2),
                                            strides = (1,1),
                                            activation='relu',
                                            data_format="channels_first")
        self.MaxPooling_3 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        #The layers to process our number
        self.Number_dense_1 = keras.layers.Dense(units=128, activation="relu")
        self.Number_dense_2 = keras.layers.Dense(units=128, activation="relu")

        #our combined layers
        self.Combined_dense_1 = keras.layers.Dense(units=128, activation="relu")
        self.Combined_dense_2 = keras.layers.Dense(units=64, activation="relu")
        self.Combined_dense_3 = keras.layers.Dense(units=64, activation="relu")
        self.Combined_dense_4 = keras.layers.Dense(units=1, activation="linear")
        """

    def call(self, input_image,input_number):#need to change to GPU
    
        #Image model
        I = self.model1(input_image)
        N = self.model2(input_number)
        x = self.model3(concat([N,I],1))
        """
        I = self.Padding_1(input_image)
        I = self.Conv2D_1(I)
        I = self.MaxPooling_1(I)
        I = self.Padding_2(I)
        I = self.Conv2D_2(I)
        I = self.MaxPooling_2(I)
        I = self.Conv2D_3(I)
        I = self.MaxPooling_3(I)
        #Flatten I so we can merge our data.
        I = keras.layers.Flatten()(I)

        #Number model
        N = self.Number_dense_1(input_number)
        N = self.Number_dense_2(N)

        #Combined model
        x = concat([N,I],1) #Concatenate through axis #1
        x = self.Combined_dense_1(x)
        x = self.Combined_dense_2(x)
        x = self.Combined_dense_3(x)
        x = self.Combined_dense_4(x)
        """
        return x
    
def update(model, optimizer, p):
    batch_size = 64
    idx = sample()
    loss_tol = 0
    statesI = tf.convert_to_tensor(np.array([list(img_buf[p][e]) for e in idx]), dtype=tf.float32)
    statesN = tf.convert_to_tensor(np.array([list(num_buf[p][e]) for e in idx]),dtype=tf.float32)
    actions=tf.convert_to_tensor(np.array([act_buf[p][e] for e in idx]),dtype=tf.float32)           
    rewards = tf.convert_to_tensor(np.array([rew_buf[p][e] for e in idx]),dtype=tf.float32)        
    done_vals = tf.convert_to_tensor(np.array([done_buf[p][e] for e in idx]),dtype=tf.float32)       
    next_statesI = tf.convert_to_tensor(np.array([list(imgN_buf[p][e]) for e in idx]),dtype=tf.float32)
    next_statesN = tf.convert_to_tensor(np.array([list(numN_buf[p][e]) for e in idx]),dtype=tf.float32)
    actions2 = tf.argmax(model[0](next_statesI, next_statesN), axis = -1)
    weights = tf.convert_to_tensor(np.array([SumSegmentTree[p][e] for e in idx]),dtype=tf.float32)
    
    y_values = model[1](next_statesI, next_statesN)
    y_values = tf.gather_nd(y_values, tf.stack([tf.range(y_values.shape[0]),
                                                    tf.cast(actions2, tf.int32)], axis=1))
    
    y_values = rewards + gamma * y_values * (1-done_vals)
    with tf.GradientTape() as tape:
        q_values = model[0](statesI, statesN)
        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                    tf.cast(actions, tf.int32)], axis=1))
        ele_loss = (y_values - q_values) ** 2
        
        weights = (cur_size * (weights/SumSegmentTree[p].sum())) ** (-beta)
        base = (cur_size * (MinSegmentTree[p].min()/SumSegmentTree[p].sum())) ** (-beta)
        weights /= base
        
        loss = tf.math.reduce_mean(ele_loss * weights)
        #loss = loss_function(y_values, q_values)
    
    grads = tape.gradient(loss, model[0].trainable_variables)
    optimizer.apply_gradients(zip(grads, model[0].trainable_variables))

    for target_weights, q_net_weights in zip(model[1].weights, model[0].weights):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)
        loss_tol += loss.numpy()
        
    updated_probs = (y_values - q_values) + 0.000001
    for id, prob in zip(idx, updated_probs):
        SumSegmentTree[p][id] = prob ** alpha
        MinSegmentTree[p][id] = prob ** alpha
    
    return loss_tol

    """
    loss_array = []
    for epoch in range(epochs):
        loss = train_step(model, optimizer,loss_function,input_images,input_numbers,labels)
        loss_array.append(loss)
    """

def get_action(network, epsilon, state, enviornment):
    if random.random() > epsilon:
        return np.argmax(network(np.expand_dims(enviornment, 0), np.expand_dims(state, 0)).numpy())
    return random.choice([0, 1, 2, 3])

def get_valid(arr, px, py):
    ret = []
    for i in range(4):
        x = px + direct[i][0]
        y = py + direct[i][1]
        if x >= 0 and x < wid and y >= 0 and y < wid:
            if arr[int(x)][int(y)][1] == 0:
                ret.append(i)
    return ret

def sample():
    indices = np.random.choice(a = range(cur_size), size = 64, p = values/sum_tree.sum())
    return indices



#the run starts here

optimizer = keras.optimizers.legacy.Adam(0.0001)
loss_function = keras.losses.MeanSquaredError()
direct = [[-1, 0], [0, 1], [1, 0], [0, -1]]
epsilon_max = 1.0 #q network, then target q network
e_decay = 0.995
epsilon_min = 0.05
gamma = 0.997
TAU = 0.5

agents = [[model(),model()],[model(),model()],[model(),model()],[model(),model()]]
agents[0][0].set_weights(agents[0][1].get_weights())
agents[1][0].set_weights(agents[1][1].get_weights())
agents[2][0].set_weights(agents[2][1].get_weights())
agents[3][0].set_weights(agents[3][1].get_weights())

number = 1
counts = 0
maxlen = 20000
wid = 16
runner = MowerEnv2([wid, number], True) #set to true later
running_avg = 0
loss_avg = 0



#PER definitions
tree_size = 1
while tree_size < maxlen: #maxlen from buffer
    tree_size *= 2
values = np.zeros(tree_size)
sum_tree = [SumSegmentTree(tree_size), SumSegmentTree(tree_size), SumSegmentTree(tree_size), SumSegmentTree(tree_size)]
min_tree = [MinSegmentTree(tree_size),MinSegmentTree(tree_size),MinSegmentTree(tree_size),MinSegmentTree(tree_size)]
cur_idx = 0
cur_size = 0
alpha = 0.6
beta = 0.4
img_buf = np.zeros((number,maxlen, 16, 16, 2))
num_buf = np.zeros((number,maxlen, 10))
rew_buf = np.zeros((number,maxlen))
act_buf = np.zeros((number,maxlen))
done_buf = np.zeros((number,maxlen))
imgN_buf = np.zeros((number,maxlen, 16, 16, 2))
numN_buf = np.zeros((number,maxlen, 10))
max_prio = [1.0, 1.0, 1.0, 1.0]
frames_tol = 100000
cur_frame = 1.0



for i in range(5000): #number of episodes
    runner.reset()
    counter = 0
    done = False
    updated = False
    tol = 0
    
    fraction = min(cur_frame/frames_tol, 1.0)
    beta = beta + fraction * (1.0 - beta)
    grid_state, agent_state = runner.get_state()
    grid_state = np.copy(grid_state)   
    agent_state = np.copy(agent_state)    
    while not done:
        actions = []
        for j in range(number): #change after done what happens later
            actions.append(get_action(agents[j][0], epsilon_max, agent_state[j], grid_state))
        reward, done= runner.step(actions)
        next_grid_state, next_agent_state = runner.get_state()
        next_grid_state = np.copy(next_grid_state)   
        next_agent_state = np.copy(next_agent_state) 
        tol += sum(reward) #change later
        done = int(done)
        
        #buffer appending
        img_buf[cur_idx] = grid_state
        num_buf[cur_idx] = agent_state
        act_buf[cur_idx] = actions
        rew_buf[cur_idx] = reward
        done_buf[cur_idx] = done
        imgN_buf[cur_idx] = next_grid_state
        numN_buf[cur_idx] = next_agent_state
        cur_idx = (cur_idx + 1) % maxlen
        cur_size = max(cur_size+1, maxlen)
        
        runner.render()
        counter += 1
        if counter % 4 == 0 and cur_size >= 2000: #every 4 to not speed up program
            for j in range(number):
                update(agents[j], optimizer, j)
            counts += 1
            updated = True
        cur_frame += 1
        grid_state = next_grid_state
        agent_state = next_agent_state 
        
    #update epsilon
    running_avg += tol
    #epsilon_max = max(epsilon_min, epsilon_max - (epsilon_max - epsilon_min) * e_decay)
    epsilon_max = max(epsilon_min, epsilon_max - 0.005)
    if updated and i % 4 == 0:
        print(running_avg/(i+1))
        print(loss_avg/(counts))
        print('------------------')

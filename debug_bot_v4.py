import pygame
import numpy as np
#from stable_baseline3.stable_baselines3.ppo import ppo
from tensorflow import keras
import tensorflow as tf
from tensorflow import concat
import time
from collections import deque
#import tensorflow_probability as tfp
import tensorflow.keras.losses as kls

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
        done = False
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
                    #if not done:
                    #    reward = self.max_area * 20
                    #else:
                    #    reward = 0
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
        if self.finished():
            return [0, 1, 2, 3]
        # Next action:
        # (feed the observation to your agent here)
        calc = np.zeros((12, 12))
        list = deque()
        calc[posx][posy] = 1
        list.append([posx, posy])
        move = []
        while len(list) > 0: #has multiple paths issues, but should be irrelevent
            size = len(list)
            for z in range(size):
                tuple = list.popleft()
                for i in range(4):
                    x = tuple[0] + self.direct[i][0]
                    y = tuple[1] + self.direct[i][1]
                    if x >= 0 and x < 12 and y >=0 and y < 12:
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
    def check_valid(self):
        calc = np.zeros((12, 12), dtype = int)
        list = deque()
        calc[0][0] = 1
        list.append([0, 0])
        while len(list) > 0: #has multiple paths issues, but should be irrelevent
            tuple = list.popleft()
            for i in range(4):
                x = tuple[0] + self.direct[i][0]
                y = tuple[1] + self.direct[i][1]
                if x >= 0 and x < 12 and y >=0 and y < 12:
                    if self.state[x][y][1] == 0 and calc[x][y] == 0:
                        calc[x][y] = 1
                        list.append((x, y))
        return (np.sum(calc) == int(self.len * self.len - np.sum(self.state)))
            
        
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
        self.area_clear = 0
        #self.max_area = np.sum(grass_data)
        #self.state = np.copy(combined_data)
        #new random generation of states
        self.max_area = 72 #change later
        self.state = np.copy(combined_data)
        #self.state = np.zeros((self.len, self.len, 2), dtype = int)
        #randomly create the walls
        """
        while True:
            for i in range(0, self.len, 2):
                for j in range(0, self.len, 2):
                    if i == 0 and j == 0:
                        continue
                    x = i + int(np.random.random() * 2)
                    y = j + int(np.random.random() * 2)
                    self.state[x][y][1] = 1
            if self.check_valid() == True:
                break
            self.state = np.zeros((self.len, self.len, 2), dtype = int)
        #randomly set the unclean blocks (half the size of grid)
        counter_a = 0
        while counter_a < self.max_area: #fix this by making it faster somehow
            x = int(np.random.random() * self.len)
            y = int(np.random.random() * self.len)
            if x < 2 and y < 2:
                continue
            if self.state[x][y][1] == 1 or self.state[x][y][0] == 1:
                continue
            self.state[x][y][0] = 1
            counter_a += 1
        """
        self.all = []
        for i in range(self.agents):
            self.all.append(np.zeros(10, dtype = int))

        for i in range(self.agents):# Set position
            self.all[i][0] = corners[i][0]
            self.all[i][1] = corners[i][1]

            # Set border detect 0 = blocked, 1 = valid
            for j in range(4):
                self.all[i][j+2] = self.isValid(self.all[i][0] + self.direct[j][0], self.all[i][1] + self.direct[j][1])
            temp = self.cheats(self.all[i][0],self. all[i][1])
            self.all[i][6] = self.all[i][7]= self.all[i][8]= self.all[i][9] = 0
            for p in temp:
                self.all[i][6+p] = 1

        self.running_length = ((self.len) * (self.len) * (1 + self.agents)) // 2
        return self.state, {}

    def get_state(self):
        return (self.state, self.all)

    def close(self):
        pass

grass_data = [[0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
                    [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                    [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                    [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                    [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0],
                    [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                    [1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                    [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1]]
walls_data =   [[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
corners = [[0, 0],[1, 0], [0, 1], [1, 1]]
combined_data = []
for i in range(12):
    combined_data.append([])
    for j in range(12):
        combined_data[i].append([grass_data[i][j], walls_data[i][j]])
combined_data = np.array(combined_data, dtype = int)
#@title

class critic(keras.Model):
    def __init__(self):
        super().__init__()
        self.m1 = keras.Sequential([
            keras.layers.Input((12, 12, 2)), #change later
            keras.layers.ZeroPadding2D(padding = (1, 1), data_format="channels_last"),
            keras.layers.Conv2D(filters = 32,kernel_size = (3,3),strides = (1,1),
activation='relu',data_format="channels_last"),
            keras.layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last"),
            keras.layers.ZeroPadding2D(padding = (1, 1), data_format="channels_last"),
            keras.layers.Conv2D(filters = 32,kernel_size = (3,3),strides = (1,1),
activation='relu',data_format="channels_last"),
            keras.layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last"),
            keras.layers.Flatten(data_format="channels_last"),
            keras.layers.Dense(units=64, activation="relu")
        ])
        self.m2 = keras.Sequential([
            keras.layers.Input((30,)),
            keras.layers.Dense(units=64, activation = 'relu')
        ])
        self.m3 = keras.Sequential([
            keras.layers.Input((128,)),
            keras.layers.Dense(units=64, activation="relu"),
            keras.layers.Dense(units=64, activation="relu"),
            keras.layers.Dense(units=1, activation="linear")
        ])

    def call(self, input_image, input_data):
        I = self.m1(input_image)
        D = self.m2(input_data)
        O = self.m3(concat([I,D],1))
        return O


class actor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.m1 = keras.Sequential([
            keras.layers.Input((12, 12, 2)), #change later
            keras.layers.ZeroPadding2D(padding = (1, 1), data_format="channels_last"),
            keras.layers.Conv2D(filters = 32,kernel_size = (3,3),strides = (1,1),
activation='relu',data_format="channels_last"),
            keras.layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last"),
            keras.layers.ZeroPadding2D(padding = (1, 1), data_format="channels_last"),
            keras.layers.Conv2D(filters = 32,kernel_size = (3,3),strides = (1,1),
activation='relu',data_format="channels_last"),
            keras.layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last"),
            keras.layers.Flatten(data_format="channels_last"),
            keras.layers.Dense(units=64, activation="relu")
        ])
        self.m2 = keras.Sequential([
            keras.layers.Input((30,)),
            keras.layers.Dense(units=64, activation = 'relu')
        ])
        self.m3 = keras.Sequential([
            keras.layers.Input((128,)),
            keras.layers.Dense(units=64, activation="relu"),
            keras.layers.Dense(units=64, activation="relu"),
            keras.layers.Dense(units=4, activation="softmax")
        ])

    def call(self, input_image, input_data):
        I = self.m1(input_image)
        D = self.m2(input_data)
        O = self.m3(concat([I,D],1))
        return O
"""
class critic(keras.Model):
    def __init__(self):
        super().__init__()
        self.m1 = keras.Sequential([
            keras.layers.Input((16, 16, 2)), #change later
            keras.layers.ZeroPadding2D(padding = (1, 1), data_format="channels_last"),
            keras.layers.Conv2D(filters = 16,kernel_size = (3,3),strides = (1,1),
activation='relu',data_format="channels_last", kernel_regularizer=keras.regularizers.L2(0.01)),
            keras.layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last"),
            tf.keras.layers.Dropout(.2),
            keras.layers.ZeroPadding2D(padding = (1, 1), data_format="channels_last"),
            keras.layers.Conv2D(filters = 16,kernel_size = (3,3),strides = (1,1),
activation='relu',data_format="channels_last", kernel_regularizer=keras.regularizers.L2(0.01)),
            keras.layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last"),
            tf.keras.layers.Dropout(.2),
            keras.layers.Flatten(data_format="channels_last"),
            keras.layers.Dense(units=64, activation="relu", kernel_regularizer=keras.regularizers.L2(0.01))
        ])
        self.m2 = keras.Sequential([
            keras.layers.Input((10,)),
            keras.layers.Dense(units=64, activation = 'relu', kernel_regularizer=keras.regularizers.L2(0.01))
        ])
        self.m3 = keras.Sequential([
            keras.layers.Input((128,)),
            keras.layers.Dense(units=64, activation="relu", kernel_regularizer=keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(.2),
            keras.layers.Dense(units=64, activation="relu", kernel_regularizer=keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(.2),
            keras.layers.Dense(units=1, activation="linear", kernel_regularizer=keras.regularizers.L2(0.01))
        ])

    def call(self, input_image, input_data):
        I = self.m1(input_image)
        D = self.m2(input_data)
        O = self.m3(concat([I,D],1))
        return O


class actor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.m1 = keras.Sequential([
            keras.layers.Input((16, 16, 2)), #change later
            keras.layers.ZeroPadding2D(padding = (1, 1), data_format="channels_last"),
            keras.layers.Conv2D(filters = 16,kernel_size = (3,3),strides = (1,1),
activation='relu',data_format="channels_last", kernel_regularizer=keras.regularizers.L2(0.01)),
            keras.layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last"),
            tf.keras.layers.Dropout(.2),
            keras.layers.ZeroPadding2D(padding = (1, 1), data_format="channels_last"),
            keras.layers.Conv2D(filters = 16,kernel_size = (3,3),strides = (1,1),
activation='relu',data_format="channels_last", kernel_regularizer=keras.regularizers.L2(0.01)),
            keras.layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last"),
            tf.keras.layers.Dropout(.2),
            keras.layers.Flatten(data_format="channels_last"),
            keras.layers.Dense(units=64, activation="relu", kernel_regularizer=keras.regularizers.L2(0.01))
        ])
        self.m2 = keras.Sequential([
            keras.layers.Input((10,)),
            keras.layers.Dense(units=64, activation = 'relu', kernel_regularizer=keras.regularizers.L2(0.01))
        ])
        self.m3 = keras.Sequential([
            keras.layers.Input((128,)),
            keras.layers.Dense(units=64, activation="relu", kernel_regularizer=keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(.2),
            keras.layers.Dense(units=64, activation="relu", kernel_regularizer=keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(.2),
            keras.layers.Dense(units=4, activation="softmax", kernel_regularizer=keras.regularizers.L2(0.01))
        ])

    def call(self, input_image, input_data):
        I = self.m1(input_image)
        D = self.m2(input_data)
        O = self.m3(concat([I,D],1))
        return O
"""

class agent():
    def __init__(self):
        self.a_opt = keras.optimizers.Adam(learning_rate=0.00005) # or 0.00015
        self.c_opt = keras.optimizers.Adam(learning_rate=0.00005)
        self.actor = actor()
        self.critic = critic()
        self.clip_pram = 0.2

    def act(self, stateI, stateN):
        prob = self.actor(stateI, stateN)
        prob = prob.numpy()
        #dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        #action = dist.sample()
        #return int(action.numpy()[0])
        return np.random.choice(a = [0, 1, 2, 3], size = 1, p = prob.flatten())

    def learn(self, statesI, statesN, actions,  adv , old_probs, discnt_rewards):
        #old_p = old_probs
        discnt_rewards = tf.convert_to_tensor(discnt_rewards)
        adv = tf.convert_to_tensor(adv)
        old_probs = tf.convert_to_tensor(old_probs)
        #old_p = tf.reshape(old_p, (len(old_p),2))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(statesI, statesN, training=True)
            v =  self.critic(statesI, statesN,training=True)
            #v = tf.reshape(v, (len(v),))
            #td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, tf.squeeze(v))
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)

        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        
        new_p = self.actor(statesI, statesN)
        p_all = tf.math.divide(new_p, p)
        kl_avg = tf.math.reduce_sum(tf.math.abs(tf.math.reduce_sum(tf.math.log(p_all), 1))).numpy() / p_all.shape[0]
        
        return a_loss, c_loss, kl_avg
    def actor_loss(self, probs, actions, adv, old_probs, closs):

        probability = probs
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        #print(probability)
        #print(entropy)
        sur1 = []
        sur2 = []
        
        for pb, t, op, a in zip(probability, adv, old_probs, actions):
                        t =  tf.constant(t)
                        #op =  tf.constant(op)
                        #print(f"t{t}")
                        #ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
                        ratio = tf.math.divide(pb[a],op[a])
                        #print(f"ratio{ratio}")
                        s1 = tf.math.multiply(ratio,t)
                        #print(f"s1{s1}")
                        s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram),t)
                        #print(f"s2{s2}")
                        sur1.append(s1)
                        sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)

        #closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.0001 * entropy)
        #print(loss)
        return loss
    
def preprocess1(reward, done, value, gamma):
    g = 0
    lmbda = 0.95
    returns = []

    for i in reversed(range(len(reward))):
        delta = reward[i] + gamma * value[i + 1] * done[i] - value[i]
        g = delta + gamma * lmbda * done[i] * g
        returns.append(g + value[i])

    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - value[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    returns = np.array(returns, dtype=np.float32)
    return returns, adv

tf.random.set_seed(369)
agent_list = [agent(),agent(),agent(),agent()]

steps = 10000
number = 3
wid = 12
que = deque(maxlen=25)
lengths = deque(maxlen=25)
clean = deque(maxlen=25)
env = MowerEnv2([wid, number], False)

"""
avg_reward = 0
min_length = 1000
avg_grass = 0
count = 0
for s in range(20):
    positions = [[0, 0],[1, 0], [0, 1], [1, 1]]
    env.reset()
    tol = 0
    done = False
    z_counter = 0
    z_grass = 0
    cur_step = 0
    dis = 1.0
    while not done:
        actions = []
        for i in range(number):
            temp = np.array([0, 0, 0, 0])
            for p in env.cheats(positions[i][0], positions[i][1]):
                temp[p] = 1
            index = np.random.choice([0, 1, 2, 3], p = (temp)/np.sum(temp))
            actions.append(index)
            positions[i][0] += env.direct[index][0]
            positions[i][1] += env.direct[index][1]
        reward, done = env.step(actions)
        for i in range(number):
            reward[i] *= dis
        dis *= 0.995
        env.render()
        pygame.time.wait(200)
        tol += sum(reward)
        cur_step += 1
        for i in range(number):
            if reward[i] > 0:
                avg_grass += 1
    avg_reward += tol
    
    count += 1
    min_length = min(min_length, cur_step)
        
print(count)
print(avg_grass/20)
print(min_length)
print(avg_reward/20)
"""

    
        
for s in range(steps):
    #if target == True:
    #    break
    total_rew = 0
    grass = 0
    done = False
    rewards = []
    statesI = []
    statesN = []
    actions = []
    probs = []
    dones = []
    values = []
    for j in range(number):
        rewards.append([])
        actions.append([])
        probs.append([])
        values.append([])

    env.reset()
    i = 0
    while True:
        stateI, tempN = env.get_state()
        stateI = tf.expand_dims(tf.convert_to_tensor(np.copy(stateI)), axis = 0)
        stateN = []
        for j in range(number):
            stateN.append(tf.expand_dims(tf.convert_to_tensor(np.copy(tempN[j])), axis = 0))
        stateN = tf.concat(stateN, 1)
        statesI.append(stateI)
        for j in range(number):
            action = agent_list[j].act(stateI, stateN)[0]
            actions[j].append(action)
            value = agent_list[j].critic(stateI, stateN).numpy()[0]
            values[j].append(value)
            prob = agent_list[j].actor(stateI, stateN).numpy()[0]
            probs[j].append(prob)
        statesN.append(stateN)
        
        reward, done= env.step([actions[j][i] for j in range(number)])
        #env.render() #rendering is here
        for j in range(number):
            rewards[j].append(reward[j])
            if reward[j] > 0:
                grass += 1
            total_rew += reward[j]
            #actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())

        dones.append(1-done)
        i += 1
        if done:
            break


    stateI, tempN = env.get_state()
    stateI = tf.expand_dims(tf.convert_to_tensor(np.copy(stateI)), axis = 0)
    stateN = []
    for j in range(number):
        stateN.append(tf.expand_dims(tf.convert_to_tensor(np.copy(tempN[j])), axis = 0))
    stateN = tf.concat(stateN, 1)
    for j in range(number):
        value = agent_list[j].critic(stateI, stateN).numpy()[0]
        values[j].append(value)
    #np.reshape(probs, (len(probs),2))
    #probs = np.stack(probs, axis=0)
    rets = []
    advs = []
    for j in range(number):
        returns, adv  = preprocess1(rewards[j], dones, values[j], 0.995)
        rets.append(returns)
        advs.append(adv)
        
        
    statesI = tf.concat(statesI, 0)
    statesN = tf.concat(statesN, 0)
    #for epocs in range(5):
    for epocs in range(4):
        cur_kl= 0
        for j in range(number):
            al,cl,kl = agent_list[j].learn(statesI, statesN, actions[j], advs[j], probs[j], rets[j])  
            cur_kl += kl

        if cur_kl > 0.0975:
            break
        
        # print(f"al{al}")
        # print(f"cl{cl}")
    que.append(total_rew)
    lengths.append(i)
    clean.append(grass)
    if s % 4 == 0 and s > 0:
        print(f"{s} rew_means: {sum(que)/len(que)}")
        print(f"{s} len_means: {sum(lengths)/len(lengths)}")
        print(f"{s} grass_means: {sum(clean)/len(clean)}")


env.close()

###########################################################
###     This file contains an implementation of         ###
###     the DQN-Algorithm first published by            ###
###     Mnih et al. in Dec. 2013.                       ###
###########################################################
###     This implementation is written by Arne Koeller  ###
###########################################################
###     References that have been used:                 ###
###     Mnih et al. (2013)                              ###
###     https://github.com/rlcode/reinforcement-learning ##
###########################################################

from numpy.random import seed
seed(26042019)
from tensorflow import set_random_seed
set_random_seed(19121909)

from util import DatasetBuilder
from util import ExperienceReplayBuffer

import sys
from gym_torcs import TorcsEnv

import random
import numpy as np

from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.models import Sequential, Model, load_model

import matplotlib.pyplot as plt



class DQNAgent:
    def __init__(self, env, track, num_of_episodes=650):

        self.env = env
        self.track = track

        ### episodes and steps
        self.max_episodes = num_of_episodes
        self.max_steps = 4000
        
        self.save_model = True
        self.load_model = False
        self.restart_memory_leak = 25

        ### size of action- and state space
        self.state_size = 70
        self.action_size_steering = 5
        self.action_size_acc = 8
        self.action_size_brake = 3

        ### DQN Hyperparameters
        self.epsilon = 1.0
        self.epsilon_decay = 1/6500
        self.epsilon_min = 0.09
        self.batch_size = 128
        self.gamma = 0.99
        self.lr = 0.002

        ### use replay memory for mini batch updates
        self.memory = ExperienceReplayBuffer(50000)

        ###
        self.model = self.build_dqn_model()
        self.target_model = self.build_dqn_model()
        self.update_target_model()
        
        ### helper class to build state representation
        self.dataset_builder = DatasetBuilder()

        

    def build_dqn_model(self):

        s = Input(shape=[self.state_size])  
        w1 = Dense(400, activation='relu')(s)
        w2 = Dense(600, activation='relu')(w1) 
        w3 = Dense(600, activation='relu')(w2) 
        steering_out = Dense(self.action_size_steering, activation='linear')(w3)
        acc_out = Dense(self.action_size_acc, activation='linear')(w3)
        brake_out = Dense(self.action_size_brake, activation='linear')(w3)

        model = Model(inputs=[s],outputs=[steering_out, acc_out, brake_out])
        adam = Adam(lr=self.lr)

        model.compile(loss='mse', optimizer=adam)
        return model

    ### update target model which is the policy the algorithm is aiming to learn
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def saveModel(self):
        self.model.save("./dqn_weights/dqn_model.h5") 

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = self.env.action_space.sample()
            steering = action[0]
            throttle = action[1]
            brake = action[2]
            

            if steering >= -1 and steering < -0.5:
                    st = -0.5
            elif steering >= -0.5 and steering < -0.3:
                    st = -0.3
            elif steering >= -0.3 and steering < 0.3:
                    st = 0
            elif steering >= 0.3 and steering < 0.5:
                    st = 0.3
            elif steering >= 0.5 and steering <= 1:
                    st = 0.5

            if throttle >= -1 and throttle < 0:
                    th = 0
            elif throttle >= 0 and throttle < 0.4:
                    th = 0.2
            elif throttle >= 0.4 and throttle < 0.5:
                    th = 0.45
            elif throttle >= 0.5 and throttle < 0.6:
                    th = 0.55
            elif throttle >= 0.6 and throttle < 0.7:
                    th = 0.65
            elif throttle >= 0.7 and throttle < 0.8:
                    th = 0.75
            elif throttle >= 0.8 and throttle < 0.9:
                    th = 0.85
            elif throttle >= 0.9 and throttle <= 1:
                    th = 0.95

            if brake >= -1 and brake < 0.3:
                    br = 0
            elif brake >= 0.3 and brake < 0.5:
                    br = 0.3
            elif brake >= 0.5 and brake <= 1:
                    br = 0.5


            
            print("Random Action!")
            
        else:
           
            q_value = self.model.predict(state)
           
            steering = np.argmax(q_value[0])
            throttle = np.argmax(q_value[1])
            brake = np.argmax(q_value[2])
            
            if steering == 0: 
                    st = -0.5
            elif steering == 1:
                    st = -0.3
            elif steering == 2:
                    st = 0
            elif steering == 3:
                    st = 0.3
            elif steering == 4:
                    st = 0.5

            if throttle == 0: 
                    th = 0
            elif throttle == 1:
                    th = 0.2
            elif throttle == 2:
                    th = 0.45
            elif throttle == 3:
                    th = 0.55
            elif throttle == 4:
                    th = 0.65
            elif throttle == 5:
                    th = 0.75
            elif throttle == 6:
                    th = 0.85
            elif throttle == 7:
                    th = 0.95

            if brake == 0: 
                    br = 0
            elif brake == 1:
                    br = 0.3
            elif brake == 2:
                    br = 0.5


            print("Calc Action!")

        ### concept of stochastic brake (described in util.py)
        if random.random() <= min(0.12, self.epsilon):
            br = 0.3
        else: 
            br = 0
            
        return [st, th, br]

    def lowerExploration(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def trainAgent(self):
        all_total_rewards = []
        all_dist_raced = []
        all_dist_percentage = []
        all_avg_speed = []

        for e in range(0, self.max_episodes):

            ### save weights every 10th episode
            if self.save_model:
                if (e % 10) == 0:
                    self.saveModel()

            ### relaunch torcs every 10th episode because 
            ### leaky memory would otherwise slow thread down 
            if (e % self.restart_memory_leak) == 0: 
                state = self.env.reset(relaunch=True) 
            else:
                state = self.env.reset()

            
            total_reward = 0
            avg_speed = 0

            state, _ = self.dataset_builder.buildStateDataSet(s=state)

            for j in range(0, self.max_steps):

                
                action = self.get_action(state.reshape(1,state.shape[0]))
                
                
                next_state, reward, done, info = self.env.step(action)
                speedX = next_state.speedX
                next_state, dist_raced = self.dataset_builder.buildStateDataSet(s=next_state)

                self.memory.memorize(state, action, reward, next_state, done)
                self.lowerExploration()
                
                self.train_model()
                total_reward += reward
                avg_speed += speedX
                state = next_state

                print("episode:", e, " step:", j, "  reward:", reward, "  action:", action, "  epsilon:", self.epsilon)

                if done:
                    
                    self.update_target_model()
                    all_total_rewards.append(total_reward)
                    all_dist_raced.append(dist_raced)

                    
                    ### use track length according to chosen track
                    if self.track == "eroad":
                        track_length = 3260
                    elif self.track == "cgspeedway":
                        track_length = 2057
                    elif self.track == "forza":
                        track_length = 5784
                    

                    percentage_of_track = round(((dist_raced/track_length) * 100),0)
                    ### in case agent completed multiple laps which is likely for a well trained agent
                    if percentage_of_track > 100: percentage_of_track = 100
                    all_dist_percentage.append(percentage_of_track)

                    all_avg_speed.append((avg_speed/j))

                    break

        self.env.end()

        print("Plotting rewards!")
        plt.plot(all_total_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Ertrag")
        plt.show()
        print("Plotting distances!")
        plt.plot(all_dist_raced)
        plt.xlabel("Episode")
        plt.ylabel("Distanz von Startlinie [m]")
        plt.show()
        print("Plotting completeness!")
        plt.plot(all_dist_percentage)
        plt.xlabel("Episode")
        plt.ylabel("Vollstaendigkeit Strecke [%]")
        plt.axis([0, 350, 0, 100])
        plt.show()
        print("Plotting avg speed!")
        plt.plot(all_avg_speed)
        plt.xlabel("Episode")
        plt.ylabel("Durschn. Geschwindigkeit [km/h]")
        plt.axis([0, 350, 0, 1])
        plt.show()


    def testAgent(self):
        ### set epsilon (exploration) low
        self.epsilon = self.epsilon_min

        ### Do not save weights when testing
        ### CHANGE if you want to continously train agent
        self.save_model = False

        try:
            self.model = load_model("./dqn_weights/dqn_model.h5")
            print("Model loaded!")
        except:
            print("Model could not be loaded! Check path or train first")
            sys.exit()

        self.trainAgent()
        

    def train_model(self):

        ### only start training after memory is of certain length
        if self.memory.getSize() < self.batch_size:
            return
        
        ### get random mini batch from replay memory
        mini_batch = self.memory.sampleRandomBatch(self.batch_size)

        ### initialize empty numpy matrices
        state = np.zeros((self.batch_size, self.state_size))
        state_next = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        ### loop through batch and build "batchsize X 1" vectors 
        ### of values in replay memory to make mini batch update
        for i in range(self.batch_size):
            state[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            state_next[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        ### get action prediction from model (the one that we seek to improve)
        target = self.model.predict(state)
        ### get action prediction from target model (the one calculating the TD Error)
        target_val = self.target_model.predict(state_next)
 
        for i in range(self.batch_size):

            ### steering conversion
            if action[i][0] == -0.5:
                x = 0
            elif action[i][0] == -0.3:
                x = 1
            elif action[i][0] == 0:
                x = 2
            elif action[i][0] == 0.3:
                x = 3
            elif action[i][0] == 0.5:
                x = 4

            ### throttle conversion
            if action[i][1] == 0:
                v = 0
            elif action[i][1] == 0.2:
                v = 1
            elif action[i][1] == 0.45:
                v = 2
            elif action[i][1] == 0.55:
                v = 3
            elif action[i][1] == 0.65:
                v = 4
            elif action[i][1] == 0.75:
                v = 5
            elif action[i][1] == 0.85:
                v = 6
            elif action[i][1] == 0.95:
                v = 7

            ### brake conversion
            if action[i][2] == 0:
                b = 0
            elif action[i][2] == 0.3:
                b = 1
            elif action[i][2] == 0.5:
                b = 2
            

            ### apply bellman optimality equation
            ### q(s,a) = r + gamma * q(s',a')
            ### EXCEPT this was the last (s, a, r, s') tuple, then a'
            ### does not exist, so we just use the reward r instead
            if done[i]:
                target[0][i][x] = reward[i]
                target[1][i][v] = reward[i]
                target[2][i][b] = reward[i]
            else:
                target[0][i][x] = reward[i] + self.gamma * (np.argmax(target_val[0][i]))
                target[1][i][v] = reward[i] + self.gamma * (np.argmax(target_val[1][i]))
                target[2][i][b] = reward[i] + self.gamma * (np.argmax(target_val[2][i]))
            
        ### train model
        self.model.fit(state, target, batch_size=self.batch_size, epochs=1, verbose=0)



# ###########################################################
# ###     This file contains an implementation of         ###
# ###     the DDPG-Algorithm first published by           ###
# ###     Lillicrap et al. in Sep. 2015.                  ###
# ###########################################################
# ###     This implementation is written by Arne Koeller  ###
# ###     Associated files are ddpg_actor.py and          ###
# ###     ddpg_critic.py and util.py                      ###
# ###########################################################
# ###     References that have been used:                 ###
# ###     Lillicrap et al. (2015)                         ###
# ###     https://github.com/yanpanlau/DDPG-Keras-Torcs   ###
# ###     https://github.com/cookbenjamin/DDPG            ###
# ###########################################################

import sys
from gym_torcs import TorcsEnv
import numpy as np
import random


from keras.models import load_model

import tensorflow as tf
from keras import backend as K


from util import DatasetBuilder
from util import OU
from util import ExperienceReplayBuffer

from ddpg_actor import Actor
from ddpg_critic import Critic

import matplotlib.pyplot as plt



class DDPGAgent:

    def __init__(self, env, track, episodes=650):

        self.env = env
        self.track = track

        self.max_episodes = episodes
        self.max_steps = 3000

        self.save_model = True
        self.load_model = False

        self.restart_memory_leak = 25

        ### size of action- and state space
        self.state_size = 70
        self.action_size = 3

        ### DDPG Hyperparameters
        self.epsilon = 1.0
        self.epsilon_decay = 1/96000
        self.epsilon_min = 0.07
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.001
        self.lr_actor = 0.00011
        self.lr_critic = 0.0011

        ### set OU Process
        self.ou = OU()

        ### tf gpu and session set
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)

        ### actor, critic and replay memory
        self.actor = Actor(self.sess, self.state_size, self.action_size, self.tau, self.lr_actor)
        self.critic = Critic(self.sess, self.state_size, self.action_size, self.tau, self.lr_critic)
        self.memory = ExperienceReplayBuffer(50000)   

        ### helper class to build state representation
        self.dataset_builder = DatasetBuilder()
        

    def saveModel(self):
        self.actor.model.save("./ddpg_weights/ddpg_actor_model.h5")
        self.critic.model.save("./ddpg_weights/ddpg_critic_model.h5")

    def lowerExploration(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        
    def trainAgent(self):   

        all_total_rewards = []
        all_dist_raced = []
        all_dist_percentage = []
        all_avg_speed = []
        all_car_hits = []
        all_race_pos = []

        for e in range(self.max_episodes):

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

       
            ### build state representation 
            state, _ = self.dataset_builder.buildStateDataSet(s=state)
            

            total_reward = 0
            avg_speed = 0
            avg_racepos = 0

            damage = 0
            damage_hit_counter = 0

            for j in range(self.max_steps):
                ### initialize numpy matrices to hold action values with OU noise
                action_with_noise = np.zeros([1,self.action_size])
                noise = np.zeros([1,self.action_size])
                
                ### get action values from actor
                action = self.actor.model.predict(state.reshape(1, state.shape[0]))
                
                ###################################################################
                ###     Deriving OU-Parameters from                             ###
                ###     https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html ###
                ###     and own experiment                                      ###
                ###################################################################
                noise[0][0] = self.epsilon * self.ou.calc_noise(action[0][0],  0.0 , 0.55, 0.15)
                noise[0][1] = self.epsilon * self.ou.calc_noise(action[0][1],  0.55 , 1.00, 0.10)
                noise[0][2] = self.epsilon * self.ou.calc_noise(action[0][2], -0.1 , 1.00, 0.05)

                ###################################################################
                ### Concept of a "stochastic" break adapted and improved from   ###
                ### https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html     ###
                ### The issue is that slamming the break all the                ###
                ### time isn't adequatly represented in the                     ###
                ### reward function. Therefore we "hack" the OU-Process         ###
                ### by triggering the brake with a chance of                    ###
                ### min(0.18, self.epsilon)                                     ###
                ################################################################### 
                if random.random() <= min(0.18, self.epsilon):
                   noise[0][2] = self.epsilon * self.ou.calc_noise(action[0][2],  0.25 , 1.00, 0.10)

                ### Add OU noise to actions
                action_with_noise[0][0] = action[0][0] + noise[0][0]
                action_with_noise[0][1] = action[0][1] + noise[0][1]
                action_with_noise[0][2] = action[0][2] + noise[0][2]

                next_state, reward, done, info = self.env.step(action_with_noise[0])

                ### build state representation
                dist_raced = next_state.distRaced
                speedX = next_state.speedX
                pre_damage = damage
                damage = next_state.damage
                racePos = next_state.racePos
                next_state = np.hstack((next_state.angle, next_state.track, next_state.focus, next_state.opponents ,next_state.trackPos, next_state.speedX, next_state.speedY,  next_state.speedZ, next_state.wheelSpinVel/100.0, next_state.rpm))
    
                ### save to experience replay memory for batch selection
                self.memory.memorize(state, action_with_noise[0], reward, next_state, done)

                ### lower epsilon for less exploration
                self.lowerExploration()

                ### train the models!
                self.trainModel()

                total_reward += reward
                avg_speed += speedX
                avg_racepos += racePos

                state = next_state

                ### detect damange
                if damage - pre_damage > 0:
                    damage_hit_counter += 1
            
                print("Episode: " +  str(e) + " Step: " + str(j) + " Action: " + str(action_with_noise) + " Reward: " + str(reward) + " Epsilon: " + str(self.epsilon))

                if done:
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

                    all_car_hits.append(damage_hit_counter)
                    all_race_pos.append(int(avg_racepos/j))

                    break


            

        self.env.end()  

        ### All the plotting stuff
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
        plt.show()
        print("Plotting avg speed!")
        plt.plot(all_avg_speed)
        plt.xlabel("Episode")
        plt.ylabel("Durschn. Geschwindigkeit [km/h]")
        plt.show()
        print("Plotting car hits!")
        plt.plot(all_car_hits)
        plt.xlabel("Episode")
        plt.ylabel("Unfaelle des Fahrzeuges")
        plt.show()
        print("Plotting avg race pos!")
        plt.plot(all_race_pos)
        plt.xlabel("Episode")
        plt.ylabel("Durschn. Position")
        plt.show()


    def trainModel(self):
        
        ### get random mini batch from experience replay memory
        mini_batch = self.memory.sampleRandomBatch(self.batch_size)
                
        ### build arrays for models from mini batch
        states = np.asarray([b[0] for b in mini_batch])
        actions = np.asarray([b[1] for b in mini_batch])
        target = np.asarray([b[1] for b in mini_batch])
        rewards = np.asarray([b[2] for b in mini_batch])
        new_states = np.asarray([b[3] for b in mini_batch])
        dones = np.asarray([b[4] for b in mini_batch])
        
        ### get q values from target critic model
        ### q(s, t(s), w') in thesis
        target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])  
        
        ### iterate through minibatch, update target according to bellman eq.
        for k in range(0, len(mini_batch)):
            if dones[k]:
                target[k] = rewards[k]
            else:
                target[k] = rewards[k] + self.gamma*target_q_values[k]
        
        ### train networks
        self.critic.model.train_on_batch([states,actions], target) 
        action_gradients = self.actor.model.predict(states)
        grads = self.critic.gradients(states, action_gradients)
        self.actor.train(states, grads)

        ### soft update
        self.actor.target_train()
        self.critic.target_train()

    
    def testAgent(self):
        ### set epsilon (exploration) low
        self.epsilon = self.epsilon_min

        ### Do not save weights when testing
        ### CHANGE if you want to continuously train agent
        self.save_model = False

        try:
            self.actor.model = load_model("./ddpg_weights/ddpg_actor_model.h5")
            self.critic.model = load_model("./ddpg_weights/ddpg_critic_model.h5")
            print("Model loaded!")
        except:
            print("Model could not be loaded! Check path or train first")
            sys.exit()

        self.trainAgent()
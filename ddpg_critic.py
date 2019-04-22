from numpy.random import seed
seed(26042019)
from tensorflow import set_random_seed
set_random_seed(19121909)

import numpy as np
import keras.backend as K
import tensorflow as tf


from keras.models import Sequential
from keras.layers import Dense, Input, Activation, concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam




class Critic:
    def __init__(self, sess, state_size, action_size,  tau, lr_actor):
        
        ### sync tf session
        self.sess = sess
        K.set_session(sess)

        ### parameter called upsilon in thesis text
        self.tau = tau

        self.lr_actor = lr_actor
        self.action_size = action_size
        
        ### Create model, get model ref, get model action inp, get model state inp
        self.model, self.action, self.state = self.init_critic_model(state_size, action_size)  

        ### Create target model mainly for soft update
        self.target_model, _ , _ = self.init_critic_model(state_size, action_size)  

        ### compute gradient -> model.output is Wx+b and self.action are actions
        self.action_gradients = tf.gradients(self.model.output, self.action)  

        ### run tf session (comp graph) after initalizing all tf variables
        self.sess.run(tf.global_variables_initializer())

    def init_critic_model(self, state_size,action_dim):

        ### Here the critic neural network is build.
        ### check thesis text for explanation

        inp1 = Input(shape=[state_size])  
        inp2 = Input(shape=[action_dim], name='action')   
        layer1_state = Dense(400, activation='relu',  kernel_initializer='he_uniform')(inp1)
        layer2_state = Dense(500, activation='linear', kernel_initializer='he_uniform')(layer1_state)
        layer1_action = Dense(500, activation='linear',  kernel_initializer='he_uniform')(inp2) 
        
  
        layer_together = concatenate([layer2_state, layer1_action])
        layer3 = Dense(550, activation='relu')(layer_together)

        outp = Dense(action_dim,activation='linear')(layer3)   

        model = Model(inputs=[inp1,inp2],outputs=outp)
        adam = Adam(lr=self.lr_actor)

        model.compile(loss='mse', optimizer=adam)

        return (model, inp2, inp1)

    ### get gradients from tf 
    def gradients(self, states, actions):
        return self.sess.run(self.action_gradients, feed_dict={
            self.state: states,
            self.action: actions 
            })[0]

    ### update target critic network according to soft updates 
    def target_train(self):
        critic_target_weights = self.target_model.get_weights()
        critic_weights = self.model.get_weights()
        for i in range(0,len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

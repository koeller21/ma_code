import numpy as np

from keras.initializers import normal, identity
from keras.models import model_from_json, load_model

from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf



class Critic(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        ### Create model and target model since DDPG is off-policy
        self.model, self.action, self.state = self.init_critic_model(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.init_critic_model(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  
        self.sess.run(tf.initialize_all_variables())

    def init_critic_model(self, state_size,action_dim):

        S = Input(shape=[state_size])  
        A = Input(shape=[action_dim], name='action2')   
        w1 = Dense(400, activation='relu',  kernel_initializer='he_uniform')(S)
        a1 = Dense(500, activation='linear',  kernel_initializer='he_uniform')(A) 
        h1 = Dense(500, activation='linear', kernel_initializer='he_uniform')(w1)
  
        h2 = concatenate([h1, a1])

        h3 = Dense(550, activation='relu')(h2)
        V = Dense(action_dim,activation='linear')(h3)   
        model = Model(inputs=[S,A],outputs=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S 

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(0,len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)



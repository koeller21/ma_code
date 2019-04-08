from numpy.random import seed
seed(26042019)
from tensorflow import set_random_seed
set_random_seed(19121909)

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Sequential, Model

from keras.layers import Dense, Input, concatenate
from keras.optimizers import Adam



class Actor:
    def __init__(self, sess, state_size, action_size, tau, lr_actor):

        ### sync tf session
        self.sess = sess
        K.set_session(sess)
        
        ### parameter called upsilon in thesis text
        self.tau = tau
        self.lr_actor = lr_actor

        ### Create model and target model since DDPG is off-policy
        self.model , self.weights, self.state = self.init_actor_model(state_size, action_size)   
        ### get target model (for prediction), target weights (for soft update) and target state
        self.target_model, _ , _ = self.init_actor_model(state_size, action_size) 

        ### build tf var that holds action gradients
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])

        ### compute gradient 
        self.parameter_gradients = tf.gradients(self.model.output, self.weights, -self.action_gradient)

        ### combine gradients and weights
        gradients = zip(self.parameter_gradients, self.weights)

        ### Apply adam optimizer with gradients from zip
        self.optimizer = tf.train.AdamOptimizer(self.lr_actor).apply_gradients(gradients)

        ### run tf session after initalizing all tf variables
        self.sess.run(tf.global_variables_initializer())

    def init_actor_model(self, state_size,action_dim):

        ### Build actor neural network
        ### check thesis text for explanation

        inp = Input(shape=[state_size])   
        layer_1 = Dense(400, activation='relu')(inp)
        layer_2 = Dense(550, activation='relu')(layer_1)
        ### Using tanh activation for steering because its in range [-1;1]
        Steering = Dense(1,activation='tanh',kernel_initializer='he_uniform')(layer_2)  

        ### Using sigmoid activation for acceleration and brake because it can't be negative
        Acceleration = Dense(1,activation='sigmoid',kernel_initializer='he_uniform')(layer_2)   
        Brake = Dense(1,activation='sigmoid',kernel_initializer='he_uniform')(layer_2) 

        outp = concatenate([Steering, Acceleration, Brake])
    
        model = Model(inputs=inp,outputs=outp)
        
        return (model, model.trainable_weights, inp)

    ### run tf session to train actor
    def train(self, states, action_gradients):
        self.sess.run(self.optimizer, feed_dict={
            self.state: states,
            self.action_gradient: action_gradients 
        })

    ### update target actor network according to soft updates
    def target_train(self):
        actor_target_weights = self.target_model.get_weights()
        actor_weights = self.model.get_weights()
        for i in range(0, len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

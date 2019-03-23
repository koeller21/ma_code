import numpy as np

from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model

from keras.layers import Dense, Flatten, Input, merge, Lambda, concatenate
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K


class Actor(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        ### Create model and target model since DDPG is off-policy
        self.model , self.weights, self.state = self.init_actor_model(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.init_actor_model(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def init_actor_model(self, state_size,action_dim):

        S = Input(shape=[state_size])   
        h0 = Dense(400, activation='relu')(S)
        h1 = Dense(550, activation='relu')(h0)
        Steering = Dense(1,activation='tanh',kernel_initializer='he_uniform')(h1)  
        Acceleration = Dense(1,activation='sigmoid',kernel_initializer='he_uniform')(h1)   
        Brake = Dense(1,activation='sigmoid',kernel_initializer='he_uniform')(h1) 

        x = concatenate([Steering, Acceleration, Brake])
    
        model = Model(inputs=S,outputs=x)
        print(model.summary())
        return model, model.trainable_weights, S

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(0,len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)




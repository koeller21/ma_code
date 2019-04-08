import numpy as np
from collections import deque
import random


class DatasetBuilder:
    def buildStateDataSet(self, s):
        angle = s.angle
        focus = s.focus
        speedX = s.speedX
        speedY = s.speedY
        speedZ = s.speedZ
        opponents = s.opponents
        rpm = s.rpm
        track = s.track
        trackPos = s.trackPos
        wheelSpinVel = s.wheelSpinVel
        stack = np.hstack((angle, focus, speedX, speedY, speedZ, opponents, rpm, track, trackPos, wheelSpinVel))
       
        return (stack, s.distRaced)

class ExperienceReplayBuffer:

    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def sampleRandomBatch(self, batch_size):
        if len(self.buffer) >= batch_size:
            return random.sample(self.buffer, batch_size)
        else:
            return random.sample(self.buffer, len(self.buffer))
        
    def memorize(self, state, action, reward, new_state, done):
        self.buffer.append([state, action, reward, new_state, done])

    def getSize(self):
        return len(self.buffer)

###########################################################
###     Reference OU class:                             ###
###     https://github.com/yanpanlau/DDPG-Keras-Torcs   ###
###########################################################

class OU:
    ### Here, np.random.randn(1) is the wiener process
    def calc_noise(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)
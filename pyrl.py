###########################################################
###########################################################
###     pyrl has been created as part of my             ###
###     masterthesis in computer science at the         ###
###     Hochschule Bochum in Germany in March 2019.     ###
###                                                     ###
###     I hereby declare that the program submitted     ###
###     is my own unaided work. All direct or indirect  ###
###     sources used are acknowledged as references.    ###
###     Arne Koeller                                    ###
###########################################################
###########################################################

from numpy.random import seed
seed(26042019)
from tensorflow import set_random_seed
set_random_seed(19121909)

import sys
import argparse

from gym_torcs import TorcsEnv
from dqn import DQNAgent
from ddpg import DDPGAgent


class Pyrl:
    def __init__(self):
        pass
    
    def parseArguments(self):
        parser = argparse.ArgumentParser(description='Reinforcement Learning Algos in Python')
        parser.add_argument("algo", type=str, help="specific RL-Algorithm   [dqn | ddpg]")
        parser.add_argument("env",  type=str, help="specific RL-Environment [torcs]")
        parser.add_argument("--mode", "-m", required=False, help="specific mode [train | test]")
        parser.add_argument("--episodes", "-e", required=False, help="number of episodes, e.g. 500")
        return parser.parse_args()

def main():

    #rl = Pyrl()
    #args = rl.parseArguments()
    #print(args.algo)
    #print(args.env)
    #print(args.mode)
    #print(args.episodes)

    env = TorcsEnv(vision=False, throttle=True)    
    
    agent = DQNAgent(env, 870)
    #agent = DDPGAgent(env, 870)
    agent.trainAgent()
    #agent.testAgent()
    

if __name__ == "__main__":
    main()

    #rl_environment_to_be_used = args.env
    #rl_algorithm_to_be_used = args.algo
    #rl_mode_to_be_used = args.mode

    #print(rl_algorithm_to_be_used)
    #print(rl_environment_to_be_used)
    #print(rl_mode_to_be_used)

    #if rl_environment_to_be_used == "torcs":
        
    #else:
    #    print("This script only supports the TORCS environment!")
    #    sys.exit()

    



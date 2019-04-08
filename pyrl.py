###########################################################
###########################################################
###     pyrl has been created as part of my             ###
###     masterthesis in computer science at the         ###
###     Hochschule Bochum in March 2019.                ###
###                                                     ###
###     I hereby declare that the program submitted     ###
###     is my own unaided work. All direct or indirect  ###
###     sources used are acknowledged as references.    ###
###                                                     ###
###     Arne Koeller                                    ###
###########################################################
###########################################################

import sys


from gym_torcs import TorcsEnv
from dqn import DQNAgent
from ddpg import DDPGAgent


class Pyrl:
    def __init__(self):
        self.numOfEpisodes = 300 
        self.parseArguments()
    
    def parseArguments(self):

        ### abort if no arguments supplied
        if len(sys.argv) < 2:
            print("Syntax: python3 pyrl.py <dqn | ddpg> <train | test> <eroad | cgspeedway | forza> ...... i.e. pyrl.py dqn test cgspeedway")
            sys.exit("Syntax error")

        ### set algorithm
        if sys.argv[1] == "dqn":
            self.algorithm = "dqn"
        elif sys.argv[1] == "ddpg":
            self.algorithm = "ddpg"
        else:
            print("Syntax: python3 pyrl.py <dqn | ddpg> <train | test> <eroad | cgspeedway | forza> ...... i.e. pyrl.py dqn test cgspeedway")
            sys.exit("Syntax error")

        ### set modus
        if sys.argv[2] == "train":
            self.modus = "train"
        elif sys.argv[2] == "test":
            self.modus = "test"
        else:
            print("Syntax: python3 pyrl.py <dqn | ddpg> <train | test> <eroad | cgspeedway | forza> ...... i.e. pyrl.py dqn test cgspeedway")
            sys.exit("Syntax error")

        ### set track
        if sys.argv[3] == "eroad":
            self.track = "eroad"
        elif sys.argv[3] == "cgspeedway":
            self.track = "cgspeedway"
        elif sys.argv[3] == "forza":
            self.track = "forza"
        else:
            print("Syntax: python3 pyrl.py <dqn | ddpg> <train | test> <eroad | cgspeedway | forza> ...... i.e. pyrl.py dqn test cgspeedway")
            sys.exit("Syntax error")

    def run(self):
        ### create TORCS environment
        env = TorcsEnv(vision=False, throttle=True)   

        ### start run according to supplied arguments
        if self.algorithm == "dqn" and self.modus == "train":
            agent = DQNAgent(env, self.track, self.numOfEpisodes)
            agent.trainAgent()
        elif self.algorithm == "dqn" and self.modus == "test":
            agent = DQNAgent(env, self.track, self.numOfEpisodes)
            agent.testAgent()
        elif self.algorithm == "ddpg" and self.modus == "train":
            agent = DDPGAgent(env, self.track, self.numOfEpisodes)
            agent.trainAgent()
        elif self.algorithm == "ddpg" and self.modus == "test":
            agent = DDPGAgent(env, self.track, self.numOfEpisodes)
            agent.testAgent()
    

def main():

    rl = Pyrl()
    rl.run()
    

if __name__ == "__main__":
    main()

    



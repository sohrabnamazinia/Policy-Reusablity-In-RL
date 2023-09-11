import gym
from gym import spaces
import numpy as np
import copy
import amazonDB

class query_refine(gym.Env):
    def __init__(self):
        self.amazonDB = amazonDB()
        
    



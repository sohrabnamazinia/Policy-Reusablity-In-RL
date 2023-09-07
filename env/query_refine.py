import gym
from gym import spaces
import numpy as np
import copy

class query_refine(gym.Env):
    def __init__(self, grid_width, grid_length, reward_system, agent_position, target_position, cell_low_value, cell_high_value, 
        start_position_value, target_position_value, block_position_value, agent_position_value, gold_position_value, block_reward, target_reward, gold_k=0, gold_positions=None, block_positions=None):
    



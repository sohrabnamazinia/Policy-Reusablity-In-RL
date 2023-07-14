import gym
from gym import spaces
import numpy as np

class GridWorld(gym.Env):
    
    def __init__(self, grid_size, reward_system, agent_position, target_position, cell_low_value, cell_high_value, 
        start_position_value, target_position_value, gold_positions=None, block_positions=None):
        
        super(GridWorld, self).__init__()

        self.grid_size = grid_size
        self.agent_position = agent_position # e.g., [0, 0]
        self.start_position = agent_position # e.g., [0, 0]
        self.target_position = target_position # e.g., [4, 4]
        self.start_position_value = start_position_value
        self.reward_system = reward_system

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low = cell_low_value,
            high= cell_high_value, shape=(grid_size, grid_size))

        # Initialize the grid
        self.grid = np.zeros((grid_size, grid_size))
        self.grid[self.start_position[0]][self.start_position[1]] = start_position_value # e.g., 5
        self.grid[self.target_position[0]][self.target_position[1]] = target_position_value # e.g., 10

        # Position the golds
        if gold_positions is not None:
            for pos in gold_positions:
                self.grid[pos[0]][pos[1]] = 1

        # Position the blocks
        if block_positions is not None:
            for pos in block_positions:
                self.grid[pos[0]][pos[1]] = -1
    
    def reset(self):
        self.agent_position = self.start_position # e.g., [0, 0]
        return np.array(self.agent_position)

    def step(self, action):
        if action == 0:   # up
            self.agent_position[0] -= 1
        elif action == 1: # right
            self.agent_position[1] += 1
        elif action == 2: # down
            self.agent_position[0] += 1
        elif action == 3: # left
            self.agent_position[1] -= 1

        self.agent_position = np.clip(self.agent_position, 0, self.grid_size-1)

        reward = self._get_reward()
        done = np.array_equal(self.agent_position, self.target_position)

        return self.agent_position, reward, done, {}

    def _get_reward(self):
        
        # gold collection task
        if self.reward_system == "gold":
            cell_value = self.grid[self.agent_position[0]][self.agent_position[1]]
            if cell_value == 1:  # gold
                return 10
            elif cell_value == -1:  # block
                return -5
            return 0

        # shortest path task
        elif self.reward_system == "path":
            dist_start = np.linalg.norm(np.array(self.start_position) - np.array(self.target_position))
            dist_now = np.linalg.norm(np.array(self.agent_position) - np.array(self.target_position))
            return 1 if dist_now < dist_start else -1

        return 0

    def render(self, mode='human'):
        print(self.grid)

    def manhattan_dist(self, state1, state2):
        term1 = abs(state1.x - state2.x)
        term2 = abs(state1.y - state2.y)
        return (term1 + term2)

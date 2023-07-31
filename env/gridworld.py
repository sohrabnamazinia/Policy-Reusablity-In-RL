import gym
from gym import spaces
import numpy as np
import copy

class GridWorld(gym.Env):
    
    def __init__(self, grid_width, grid_length, reward_system, agent_position, target_position, cell_low_value, cell_high_value, 
        start_position_value, target_position_value, gold_positions=None, block_positions=None):
        
        super(GridWorld, self).__init__()

        self.grid_width = grid_width
        self.grid_length = grid_length
        self.agent_position = agent_position # e.g., [0, 0]
        self.start_position = agent_position # e.g., [0, 0]
        self.target_position = target_position # e.g., [4, 4]
        self.start_position_value = start_position_value
        self.reward_system = reward_system
        self.gold_positions = gold_positions
        self.block_positions = block_positions
        self.MINE_REWARD = -100

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low = cell_low_value,
            high= cell_high_value, shape=(self.grid_width, self.grid_length))

        # Initialize the grid
        self.grid = np.zeros((self.grid_width, self.grid_length))
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
        self.agent_position = copy.copy(self.start_position) # e.g., [0, 0]
        for gold in self.gold_positions:
            self.grid[gold[0]][gold[1]] = 1
        #return np.array(self.agent_position)
        return self.grid

    def step(self, action):
        prev_agent_position = [self.agent_position[0], self.agent_position[1]]

        if action == 0:   # up
            self.agent_position[0] -= 1
        elif action == 1: # right
            self.agent_position[1] += 1
        elif action == 2: # down
            self.agent_position[0] += 1
        elif action == 3: # left
            self.agent_position[1] -= 1

        self.agent_position = np.clip(self.agent_position, (0, 0), (self.grid_width - 1, self.grid_length - 1))

        reward = self._get_reward(prev_agent_position)
        done = np.array_equal(self.agent_position, self.target_position)


        #return self.agent_position, reward, done, {}
        return self.grid, reward, done, {}

    def _get_reward(self, prev_agent_position):
        
        # gold collection task
        if self.reward_system == "gold":
            return self.get_reward_gold()

        # shortest path task
        elif self.reward_system == "path":
            return self.get_reward_path(prev_agent_position)
        
        elif self.reward_system == "combined":
            return self.get_reward_gold() + self.get_reward_path(prev_agent_position)

        return 0

    def render(self, mode='human'):
        print(self.grid)

    def get_reward_path(self, prev_agent_position):
        current_cell_value = self.grid[self.agent_position[0]][self.agent_position[1]]
        if current_cell_value == -1:  # block
            return self.MINE_REWARD
        d1 = np.sum(np.abs(np.array(prev_agent_position) - np.array(self.target_position)))
        d2 = np.sum(np.abs(np.array(self.agent_position) - np.array(self.target_position)))
        r = d1 - d2
        return r    
        
    def get_reward_gold(self):
        reward = 0
        candidates = []
        current_cell_value = self.grid[self.agent_position[0]][self.agent_position[1]]
        if current_cell_value == -1:  # block
            return self.MINE_REWARD
        for i in range(-2, 3):
            for j in range(-2, 3):
                new_candidate = [self.agent_position[0] + i, self.agent_position[1] + j]
                new_candidate = np.clip(new_candidate, (0, 0), (self.grid_width - 1, self.grid_length - 1)).tolist()
                if new_candidate not in candidates:
                    candidates.append(new_candidate)
        for candidate in candidates:
            cell_value = self.grid[candidate[0], candidate[1]]
            if cell_value == 1:
                reward += 1
        if current_cell_value == 1:  # gold
            self.grid[self.agent_position[0]][self.agent_position[1]] = 0
        return reward

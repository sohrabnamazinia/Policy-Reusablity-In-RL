from env.gridworld import GridWorld
from agents.deep_agent import Agent
from agents.q_agent import QLearningAgent
import wandb
import numpy as np

timesteps = 10000
deep_algorithm = "DQN" # either "DQN" or "A2C"
reward_system = "gold" # either "gold" or "path"
gold_positions = [[1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [5, 4], [5, 5], [5, 6]]
block_positions = [[2, 2], [3, 4], [5, 5], [6, 6], [7, 7], [8, 8]]
agent_initial_position = [0,0]
target_position = [5,4]
cell_low_value = -1
cell_high_value = 10
start_position_value = 5
target_position_value = 10
max_steps_per_episode = 100
grid_size = 10

# Initialize a new run
run = wandb.init(project="gridworld", entity="omidvar")

# Instantiate environment
grid_world = GridWorld(grid_size= grid_size, reward_system = reward_system, 
	agent_position = agent_initial_position, target_position = target_position,
	cell_low_value = cell_low_value, cell_high_value = cell_high_value,
	start_position_value = start_position_value, target_position_value = target_position_value,
	gold_positions = gold_positions, block_positions = block_positions)

# Instantiate and train deep agent
deep_agent = Agent(grid_world, deep_algorithm)
deep_agent.learn(timesteps)
deep_agent.save("agent.pkl")

# End the run
run.finish()


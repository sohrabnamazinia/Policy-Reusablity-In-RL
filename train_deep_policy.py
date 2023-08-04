from env.gridworld import GridWorld
from agents.deep_agent import Agent
from agents.q_agent import QLearningAgent
import wandb
import numpy as np
from env.init_gridworld import init_gridworld_1

timesteps = 10000
deep_algorithm = "DQN" # either "DQN" or "A2C"
reward_system = "path" # either "gold" or "path" or "combined"


# Initialize a new run
run = wandb.init(project="gridworld", entity="sn773")

grid_world = init_gridworld_1(reward_system=reward_system)

# Instantiate and train deep agent
deep_agent = Agent(grid_world, deep_algorithm)
deep_agent.learn(timesteps)
deep_agent.save("agent.pkl")

# End the run
run.finish()


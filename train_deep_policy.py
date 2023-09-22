from agents.deep_agent import Agent
import wandb
from env.init_gridworld import init_gridworld_1
import time


def train_deep(grid_world, deep_algorithm, reward_system, timesteps, output_path):
    # Initialize a new run
    #run = wandb.init(project="gridworld", entity="sn773")
    start_time = time.time()
    # Instantiate and train deep agent
    deep_agent = Agent(grid_world, deep_algorithm)
    deep_agent.learn(timesteps)
    deep_agent.save(output_path)
    train_time = time.time() - start_time
    return train_time
    # End the run
    #run.finish()

# #input
# timesteps = 1000
# deep_algorithm = "PPO" 
# reward_system = "path" 
# output_path = "deep_" + deep_algorithm + ".pkl"
# grid_world = init_gridworld_1(reward_system=reward_system)

# train_deep(grid_world, deep_algorithm, reward_system, timesteps, output_path)

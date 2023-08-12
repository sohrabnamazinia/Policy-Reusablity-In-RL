from agents.deep_agent import Agent
import wandb
from env.init_gridworld import init_gridworld_1
import time

def train_deep_policy(timesteps, deep_algorithm, reward_system, output_path):
    #timesteps = 10000
    #deep_algorithm = "PPO" 
    #reward_system = "path" 

    # Initialize a new run
    run = wandb.init(project="Train_Deep")
    start_time = time.time()
    
    #run = wandb.init(project="gridworld", entity="sn773")
    grid_world = init_gridworld_1(reward_system=reward_system)

    # Instantiate and train deep agent
    deep_agent = Agent(grid_world, deep_algorithm)
    deep_agent.learn(timesteps)
    
    total_time = start_time - time.time()
    wandb.log({"Total Training Time": total_time}, step=1)
    
    deep_agent.save(output_path)

    # End the run
    run.finish()
    return total_time


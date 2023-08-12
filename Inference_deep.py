from agents.deep_agent import Agent
import numpy as np
from env.init_gridworld import init_gridworld_1
import wandb

# inference method
# the agent.pkl file must be in the same directory as this file
def inference_deep(grid_world, algorithm, agent_path):
    
	run = wandb.init(project="Inference_Deep")
	deep_agent = Agent(grid_world, algorithm)
	deep_agent.load(agent_path, grid_world)
	
    # Reset the environment to its initial state, and record returned observation
	obs = grid_world.reset()

    # Maximum number of steps for inference
	max_steps_inference = 100

	for step in range(max_steps_inference):

        # print the current location of the agent
		print("Agent location: " + str(grid_world.agent_position))
		
        # action selection from model (inference)
		action, _states = deep_agent.model.predict(observation = obs)

		print("Action: ", action)

        # step
		obs, reward, done, _ = grid_world.step(action)

        # print step index
		print("Step: ", step + 1)
		grid_world.render()

        # check if the agent reached the target or the maximum number of steps is reached
		if done:
			if reward > 0:
				print("Agent reached the target!")
			else:
				print("Agent failed to reach the target!")
			break
		run.finish()

reward_system = "path"	
algorithm = "PPO"	
agent_path = "agent.pkl"	
grid_world = init_gridworld_1(reward_system)
inference_deep(grid_world, algorithm, agent_path)

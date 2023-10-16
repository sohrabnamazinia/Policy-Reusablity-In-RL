from agents.deep_agent import Agent
import numpy as np
from env.init_gridworld import init_gridworld_1
import wandb
import time

# inference method
# the agent.pkl file must be in the same directory as this file
def inference_deep(grid_world, algorithm, agent_path):

	total_time = 0
	grid_world.reset()

	deep_agent = Agent(grid_world, algorithm)
	deep_agent.load(agent_path, grid_world)
	
    # Reset the environment to its initial state, and record returned observation
	obs = grid_world.reset()

    # Maximum number of steps for inference
	max_steps_inference = 100
	path = []
	cumulative_reward = 0
	path.append(grid_world.agent_position.copy())

	for step in range(max_steps_inference):
     
		start_time = time.time()
        # print the current location of the agent
		print("Agent location: " + str(grid_world.agent_position))
		
        # action selection from model (inference)
		action, _states = deep_agent.model.predict(observation = obs)
		

		print("Action: ", action)

		prev_agent_position = grid_world.agent_position.copy()

        # step
		obs, reward, done, _ = grid_world.step(action)
		if grid_world.agent_position != prev_agent_position:
			path.append(grid_world.agent_position.copy())


		# update return values
		cumulative_reward += reward

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
		elapsed_time = time.time() - start_time
		total_time += elapsed_time
		#wandb.log({"Total Inference Time": total_time}, step=step)
        
	#run.finish()
	return path, cumulative_reward, total_time

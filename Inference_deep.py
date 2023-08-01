from agents.deep_agent import Agent
import numpy as np
from env.gridworld import GridWorld
from env.init_gridworld import init_gridworld_1


# inference method
# the agent.pkl file must be in the same directory as this file

def inference_deep(grid_world):
	
	deep_agent = Agent(grid_world, "DQN")
	deep_agent.load('agent.pkl')
	
    # Reset the environment to its initial state, and record returned observation
	obs = grid_world.reset()
	state_index = np.ravel_multi_index(tuple(grid_world.agent_position), dims=grid_world.grid.shape)

    # Maximum number of steps for inference
	max_steps_inference = 100

	for step in range(max_steps_inference):

        # print the current location of the agent
		print("Agent location: " + str(grid_world.agent_position))
		
        # action selection from model (inference)
		action, _states = deep_agent.model.predict(observation = obs, deterministic=True)
		#action, _states = deep_agent.model.predict(observation = obs, state=state_index, deterministic=True)

		print("Action: ", action)

        # step
		obs, reward, done, _ = grid_world.step(action)
		next_state_index = np.ravel_multi_index(tuple(grid_world.agent_position.flatten()), dims=grid_world.grid.shape)

        # upadate state index
		state_index = next_state_index

        # print step index
		print("Step: ", step + 1)
		print(obs)

        # check if the agent reached the target or the maximum number of steps is reached
		if done:
			if reward > 0:
				print("Agent reached the target!")
			else:
				print("Agent failed to reach the target!")
			break
			
grid_world = init_gridworld_1('combined')
inference_deep(grid_world)

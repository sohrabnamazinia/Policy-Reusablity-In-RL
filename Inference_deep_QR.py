from agents.deep_agent import Agent
import time

# inference method
# the agent.pkl file must be in the same directory as this file
def inference_deep(env, algorithm, agent_path):

	total_time = 0
	env.reset()

	deep_agent = Agent(env, algorithm)
	deep_agent.load(agent_path, env)
	
    # Reset the environment to its initial state, and record returned observation
	obs = env.reset()

    # Maximum number of steps for inference
	max_steps_inference = 8
	path = []
	cumulative_reward = 0
	path.append(env.get_state_index())

	for step in range(max_steps_inference):
     
		start_time = time.time()
        # print the current location of the agent
		print("Agent state: " + str(env.get_state_index()))
		
        # action selection from model (inference)
		action, _states = deep_agent.model.predict(observation = obs)
		

		print("Action: ", action)

		prev_agent_state_index = env.get_state_index().copy()

        # step
		obs, reward, done, _ = env.step(action)
		if env.get_state_index() != prev_agent_state_index:
			path.append(env.get_state_index())


		# update return values
		cumulative_reward += reward

        # print step index
		print("Step: ", step + 1)

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

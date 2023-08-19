from train_deep_policy import train_deep_policy
from env.init_gridworld import init_gridworld_1
from Inference_deep import inference_deep
from agents.deep_agent import Agent

# Define env and train parameters
reward_system = "combined"
grid_world = init_gridworld_1(reward_system)
timesteps = 1000
deep_algorithm = "PPO"
output_path = "agent_new.pkl"

# train + inference

deep_agent = Agent(grid_world, deep_algorithm)
deep_agent.learn(timesteps)
deep_agent.save(output_path)

total_train_time = deep_agent.callback.total_time

grid_world = init_gridworld_1(reward_system)
deep_agent_path = output_path
total_inference_time = inference_deep(grid_world, deep_algorithm, deep_agent_path)
total_time = total_train_time + total_inference_time

# output
print("Deep_Algorithm: " + deep_algorithm + ", Policy: " + reward_system + ":")
print("Train+Inference time: " + str(total_time))
print("Train time: " + str(total_train_time))
print("Inference time: " + str(total_inference_time))


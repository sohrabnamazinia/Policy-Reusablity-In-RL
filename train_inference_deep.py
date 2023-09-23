from train_deep_policy import train_deep
from env.init_gridworld import init_gridworld_1
from Inference_deep import inference_deep

# Define env and train parameters
reward_system = "combined"
grid_world = init_gridworld_1(reward_system)
timesteps = 1000
deep_algorithm = "PPO"
output_path = "agent_" + deep_algorithm + ".pkl"

# train inference
total_train_time = train_deep(grid_world, deep_algorithm, timesteps, output_path)

# inference
path, cumulative_reward, total_inference_time = inference_deep(grid_world, deep_algorithm, output_path)
total_time = total_train_time + total_inference_time

# output
print("Deep_Algorithm: " + deep_algorithm + ", Policy: " + reward_system + ":")
print("Train+Inference time: " + str(total_time))
print("Train time: " + str(total_train_time))
print("Inference time: " + str(total_inference_time))
print("Path: " + str(path))
print("Cumulative reward: " + str(cumulative_reward))

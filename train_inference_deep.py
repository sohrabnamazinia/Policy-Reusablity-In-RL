from train_deep_policy import train_deep_policy
from env.init_gridworld import init_gridworld_1
from Inference_deep import inference_deep

# Define env and train parameters
reward_system = "combined"
grid_world = init_gridworld_1(reward_system)
n_episodes = 1000
max_steps_per_episode = 100
agent_type = "Sarsa"
output_path = "q_table_combined.npy"

# train + inference
total_train_time = train_q_policy(grid_world, n_episodes, max_steps_per_episode, agent_type, output_path)
grid_world = init_gridworld_1(reward_system)
q_table_path = output_path
total_inference_time = inference_q(grid_world, q_table_path)
total_time = total_train_time + total_inference_time

# output
print("Agent_Type: " + agent_type + ", Policy: " + reward_system + ":")
print("Train+Inference time: " + str(total_time))
print("Train time: " + str(total_train_time))
print("Inference time: " + str(total_inference_time))


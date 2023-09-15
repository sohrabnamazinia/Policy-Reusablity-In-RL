from env.init_gridworld import init_gridworld_4
from train_q_policy import train_q_policy
import pandas as pd
from utilities import plot_cumulative_reward_env_size

#inputs
env_test_count = 4
first_env_size = 4
env_test_step = 2
n_episodes = 1000
max_steps_per_episode = 100
result_step_size = 10
reward_system = "combined"
agent_type = "QLearning"

#output
cumulative_rewards = []
csv_file_name = "cumulative_reward_" + str(n_episodes) + ".csv"
output_path = "q_table_test_cummulative_reward_" + str(n_episodes) + ".npy"

# setup plot
header = ["Environment Size", "Cumulative Reward"]
environment_size_index = 0
cumulative_reward_index = 1
df = pd.DataFrame()

env_sizes = []
for i in range(env_test_count):
    env_width = env_test_step * (i) + first_env_size
    env_length = env_width
    env_sizes.append((env_width, env_length))


environments = []
for (env_width, env_length) in env_sizes:
    grid_world = init_gridworld_4(reward_system, env_width, env_length)
    environments.append(grid_world)

for i in range(env_test_count):
    env = environments[i]
    total_time, dag, cumulative_reward = train_q_policy(env, n_episodes, max_steps_per_episode, agent_type, output_path, result_step_size=result_step_size)
    cumulative_rewards.append(cumulative_reward)
    df.at[i, environment_size_index] = env.grid_width * env.grid_length
    df.at[i, cumulative_reward_index] = cumulative_reward

df.to_csv(csv_file_name, index=False, header=header)
print("Cumulative rewards:\n" + str(cumulative_rewards))
plot_cumulative_reward_env_size(csv_file_name, header[0], header[1])
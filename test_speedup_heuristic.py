from env.init_gridworld import init_gridworld_3
import pandas as pd
from train_q_policy import train_q_policy
from inference_q import inference_q
from pruning import run_pruning
from utilities import plot_speedup
from heuristic import run_heuristic

#inputs
env_test_count = 5
first_env_size = 4
env_test_step = 2
n_episodes = 1000
max_steps_per_episode = 100
result_step_size = 10
learning_rate = 0.1
discount_factor = 0.99
heuristic_k = 1
agent_type = "QLearning"

#output
times_train_scratch = []
times_heuristic = []
speedups = []
csv_file_name = "Test_Speedup_heuristic_" + agent_type + ".csv"

q_table_1_output_path = "q_table_1.npy"
q_table_2_output_path = "q_table_2.npy"
q_table_3_output_path = "q_table_combined.npy"

env_sizes = []
for i in range(env_test_count):
    env_width = env_test_step * (i) + first_env_size
    env_length = env_width
    env_sizes.append((env_width, env_length))

policy_1_environments = []
policy_2_environments = []
combined_environments = []
for (env_width, env_length) in env_sizes:
    grid_world_1 = init_gridworld_3("path", env_width, env_length)
    grid_world_2 = init_gridworld_3("gold", env_width, env_length)
    grid_world_3 = init_gridworld_3("combined", env_width, env_length)
    policy_1_environments.append(grid_world_1)
    policy_2_environments.append(grid_world_2)
    combined_environments.append(grid_world_3)


# setup panda
df = pd.DataFrame()
header = ["Environment Size", "Time - heuristic", "Time - train scratch", "Speedup"]
env_size_index = 0
time_heuristic_index = 1
time_train_scratch_index = 2
speedup_index = 3

for i in range(env_test_count):
    path_env = policy_1_environments[i]
    gold_env = policy_2_environments[i]
    combined_env = combined_environments[i]
    heuristic_max_allowed_path_size = combined_env.grid_width + combined_env.grid_length

    time_path, dag_path, _ = train_q_policy(grid_world=path_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_1_output_path, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
    time_gold, dag_gold, _ = train_q_policy(grid_world=gold_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_2_output_path, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
    time_train_combined, dag_combined, _ = train_q_policy(grid_world=combined_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_3_output_path, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
    inference_time_combined, reward_ground_truth, _ = inference_q(grid_world=combined_env, q_table_path=q_table_3_output_path)
    time_from_scratch = time_train_combined + inference_time_combined
    max_cumulative_reward, best_path, paths, shortest_paths, time_heuristic = run_heuristic(q_table_1_output_path, q_table_2_output_path, heuristic_k, heuristic_max_allowed_path_size, gridworld=combined_env)

    speedup = time_from_scratch / time_heuristic
    df.at[i, env_size_index] = str((combined_env.grid_width, combined_env.grid_length))
    df.at[i, time_heuristic_index] = time_heuristic
    df.at[i, time_train_scratch_index] = time_from_scratch
    df.at[i, speedup_index] = speedup
    times_train_scratch.append(time_from_scratch)
    times_heuristic.append(time_heuristic)
    speedups.append(speedup)
    df.to_csv(csv_file_name, index=False, header=header)
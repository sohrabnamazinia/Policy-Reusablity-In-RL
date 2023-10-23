from env.init_gridworld import init_gridworld_6
import pandas as pd
from train_q_policy import train_q_policy
from inference_q import inference_q
from pruning import run_pruning
from utilities import plot_speedup_action_size

#inputs
action_test_count = 4
first_action_size = 2
action_test_step = 1
n_episodes = 1000
max_steps_per_episode = 100
learning_rate = 0.1
discount_factor = 0.99
agent_type = "QLearning"

#output
times_train_scratch = []
times_ExNonZeroDiscount = []
times_greedy_k = []
speedups_ExNonZeroDiscount = []
speedups_greedy_k = []
csv_file_name = "Test_Speedup_action_size_" + agent_type + ".csv"

q_table_1_output_path = "q_table_1.npy"
q_table_2_output_path = "q_table_2.npy"
q_table_3_output_path = "q_table_combined.npy"

action_sizes = []
for i in range(action_test_count):
    action_size = action_test_step * (i) + first_action_size
    action_sizes.append(action_size)

policy_1_environments = []
policy_2_environments = []
combined_environments = []
for action_size in action_sizes:
    grid_world_1 = init_gridworld_6("path", action_size=action_size)
    grid_world_2 = init_gridworld_6("gold", action_size=action_size)
    grid_world_3 = init_gridworld_6("combined", action_size=action_size)
    policy_1_environments.append(grid_world_1)
    policy_2_environments.append(grid_world_2)
    combined_environments.append(grid_world_3)

# setup panda
df = pd.DataFrame()
header = ["Action Size", "Time - ExNonZeroDiscount", "Time - Greedy-K", "Time - train scratch", "Speedup - ExNonZeroDiscount", "Speedup - Greedy-K"]
action_size_index = 0
time_ExNonZeroDiscount_index = 1
time_greedy_k_index = 2
time_train_scratch_index = 3
speedup_ExNonZeroDiscount_index = 4
speedup_greedy_k_index = 5

for i in range(action_test_count):
    path_env = policy_1_environments[i]
    gold_env = policy_2_environments[i]
    combined_env = combined_environments[i]
    time_path, dag_path, _ = train_q_policy(grid_world=path_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_1_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
    time_gold, dag_gold, _ = train_q_policy(grid_world=gold_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_2_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
    time_train_combined, dag_combined, _ = train_q_policy(grid_world=combined_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_3_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
    inference_time_combined, reward_ground_truth, _ = inference_q(grid_world=combined_env, q_table_path=q_table_3_output_path)
    time_from_scratch = time_train_combined + inference_time_combined
    best_path, max_reward, time_ExNonZeroDiscount, pruning_percentage = run_pruning(gridworld=combined_env, dag_1=dag_path, dag_2=dag_gold, discount_factor=discount_factor, learning_rate=learning_rate)

    speedup_ExNonZeroDiscount = time_from_scratch / time_ExNonZeroDiscount
    df.at[i, action_size_index] = combined_env.action_count
    df.at[i, time_ExNonZeroDiscount_index] = time_ExNonZeroDiscount
    df.at[i, time_train_scratch_index] = time_from_scratch
    df.at[i, speedup_ExNonZeroDiscount_index] = speedup_ExNonZeroDiscount
    times_train_scratch.append(time_from_scratch)
    times_ExNonZeroDiscount.append(time_ExNonZeroDiscount)
    speedups_ExNonZeroDiscount.append(speedup_ExNonZeroDiscount)
    df.to_csv(csv_file_name, index=False, header=header)

plot_speedup_action_size(csv_file_name, header[0], header[1], header[2], header[3], header[4], header[5])
print("Action sizes: " + str(action_sizes))
print("Total times for ExNonZeroDiscount: " + str(times_ExNonZeroDiscount))
print("Total times for greedy-k algorithm: " + str(times_greedy_k))
print("Total times for Train from scratch: " + str(times_train_scratch))
print("Speedups for ExNonZeroDiscount: " + str(speedups_ExNonZeroDiscount))
print("Speedups for greedy-k: " + str(speedups_greedy_k))
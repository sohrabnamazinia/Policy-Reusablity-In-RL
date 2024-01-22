from env.init_query_refine import init_query_refine_2
import pandas as pd
from train_q_qr import train_q_qr
from inference_q_qr import inference_q_qr
from pruning_qr import run_pruning
from utilities import plot_speedup_qr
from heuristic_qr import run_heuristic

#inputs
env_test_count = 5
first_env_size = 7
env_test_step = 1
n_episodes = 1
max_steps_per_episode = 4
learning_rate = 0.1
discount_factor = 0.99
heuristic_k = 2
agent_type = "QLearning"

#output
times_train_scratch = []
times_heuristic = []
speedups = []
csv_file_name = "Test_Speedup_heuristic_" + agent_type + "_QR.csv"

q_table_1_output_path = "q_table_1.npy"
q_table_2_output_path = "q_table_2.npy"
q_table_3_output_path = "q_table_combined.npy"

env_sizes = []
for i in range(env_test_count):
    env_size = env_test_step * (i) + first_env_size
    env_sizes.append(env_size)

closeness_environments = []
feature_environments = []
combined_environments = []
for env_size in env_sizes:
    env_1 = init_query_refine_2("closeness", env_size)
    env_2 = init_query_refine_2("feature", env_size)
    env_3 = init_query_refine_2("combined", env_size)
    closeness_environments.append(env_1)
    feature_environments.append(env_2)
    combined_environments.append(env_3)


# setup panda
df = pd.DataFrame()
header = ["Environment Size", "Time - heuristic", "Time - train scratch", "Speedup"]
env_size_index = 0
time_heuristic_index = 1
time_train_scratch_index = 2
speedup_index = 3

for i in range(env_test_count):
    closeness_env = closeness_environments[i][0]
    feature_env = feature_environments[i][0]
    combined_env = combined_environments[i][0]
    heuristic_max_allowed_path_size = combined_env.state_count * combined_env.state_count 
    time_closeness, dag_closeness = train_q_qr(env=closeness_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_1_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
    time_feature, dag_feature = train_q_qr(env=feature_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_2_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
    time_train_combined, dag_combined = train_q_qr(env=combined_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_3_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
    inference_time_combined, _, _ = inference_q_qr(env=combined_env, q_table_path=q_table_3_output_path, edge_dict=dag_combined.edge_dict)
    time_from_scratch = time_train_combined + inference_time_combined
    combined_env.reset()
    max_cumulative_reward, best_path, paths, shortest_paths, time_heuristic = run_heuristic(q_table_1_path=q_table_1_output_path, q_table_2_path=q_table_2_output_path, k=heuristic_k, max_allowed_path_size=heuristic_max_allowed_path_size, dag_closeness=dag_closeness, dag_feature=dag_feature, reward_system=None, env=combined_env)
    
    speedup = time_from_scratch / time_heuristic
    df.at[i, env_size_index] = int(combined_env.state_count)
    df.at[i, time_heuristic_index] = time_heuristic
    df.at[i, time_train_scratch_index] = time_from_scratch
    df.at[i, speedup_index] = speedup
    times_train_scratch.append(time_from_scratch)
    times_heuristic.append(time_heuristic)
    speedups.append(speedup)
    df.to_csv(csv_file_name, index=False, header=header)

plot_speedup_qr(csv_file_name, header[0], header[1], header[2], header[3])
print("State space sizes: " + str(env_sizes))
print("Total times for Heuristic: " + str(times_heuristic))
print("Total times for training combined policy from scratch: " + str(times_train_scratch))
print("Speedups: " + str(speedups))
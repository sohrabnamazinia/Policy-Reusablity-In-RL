from env.init_query_refine import init_query_refine_2
import pandas as pd
from train_q_qr import train_q_qr
from inference_q_qr import inference_q_qr
from pruning_qr import run_pruning
from utilities import plot_speedup_qr

#inputs
env_test_count = 1
first_env_size = 11
env_test_step = 1
n_episodes = 4
max_steps_per_episode = 8
learning_rate = 0.1
discount_factor = 0.99
agent_type = "QLearning"

#output
times_train_scratch = []
times_ExNonZeroDiscount = []
speedups = []
csv_file_name = "Test_Speedup_QR_" + agent_type + ".csv"

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
    env_1, _ = init_query_refine_2("closeness", env_size)
    env_2, _ = init_query_refine_2("feature", env_size)
    env_3, _ = init_query_refine_2("combined", env_size)
    closeness_environments.append(env_1)
    feature_environments.append(env_2)
    combined_environments.append(env_3)


# setup panda
df = pd.DataFrame()
header = ["Environment Size", "Time - ExNonZeroDiscount", "Time - train scratch", "Speedup"]
env_size_index = 0
time_ExNonZeroDiscount_index = 1
time_train_scratch_index = 2
speedup_index = 3

for i in range(env_test_count):
    closeness_env = closeness_environments[i]
    feature_env = feature_environments[i]
    combined_env = combined_environments[i]
    time_closeness, dag_closeness = train_q_qr(env=closeness_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_1_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
    time_feature, dag_feature = train_q_qr(env=feature_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_2_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
    time_train_combined, dag_combined = train_q_qr(env=combined_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_3_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
    inference_time_combined, _, _ = inference_q_qr(env=combined_env, q_table_path=q_table_3_output_path, edge_dict=dag_combined.edge_dict)
    time_from_scratch = time_train_combined + inference_time_combined
    best_path, max_reward, time_ExNonZeroDiscount, pruning_percentage = run_pruning(env=combined_env, dag_1=dag_closeness, dag_2=dag_feature, discount_factor=discount_factor, learning_rate=learning_rate)

    speedup = time_from_scratch / time_ExNonZeroDiscount
    df.at[i, env_size_index] = int(combined_env.state_count)
    df.at[i, time_ExNonZeroDiscount_index] = time_ExNonZeroDiscount
    df.at[i, time_train_scratch_index] = time_from_scratch
    df.at[i, speedup_index] = speedup
    times_train_scratch.append(time_from_scratch)
    times_ExNonZeroDiscount.append(time_ExNonZeroDiscount)
    speedups.append(speedup)
    df.to_csv(csv_file_name, index=False, header=header)

plot_speedup_qr(csv_file_name, header[0], header[1], header[2], header[3])
print("State space sizes: " + str(env_sizes))
print("Total times for ExNonZeroDiscount: " + str(times_ExNonZeroDiscount))
print("Total times for training combined policy from scratch: " + str(times_train_scratch))
print("Speedups: " + str(speedups))
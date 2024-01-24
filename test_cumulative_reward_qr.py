from env.init_query_refine import init_query_refine_2
from train_q_qr import train_q_qr
from inference_q_qr import inference_q_qr
from pruning_qr import run_pruning
import pandas as pd
from utilities import plot_cumulative_reward_qr
from heuristic_qr import run_heuristic


#inputs
env_test_count = 5
# 1 to 4
diff_start_query_test = 4
first_env_size = 7
env_test_step = 1
n_episodes = 1
max_steps_per_episode = 4
learning_rate = 0.1
heuristic_k = 1
discount_factor = 0.99
agent_type = "Sarsa"
_, new_queries = init_query_refine_2("closeness", first_env_size)

#output
avg_rewards_exnonzero = []
avg_rewards_train_scratch = []
avg_rewards_heuristic = []
csv_file_name = "Test_Cumulative_Reward_QR_" + agent_type + ".csv"
q_table_1_output_path = "q_table_closeness.npy"
q_table_2_output_path = "q_table_feature.npy"
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
header = ["State Space Size", "Avg Reward - ExNonZeroDiscount", "Avg Reward - Train From Scratch", "Avg Reward - Greedy-K"]
env_size_index = 0
avg_cr_exnonzero_index = 1
avg_cr_train_scratch_index = 2
avg_cr_heuristic_index = 3


for i in range(env_test_count):
    avg_reward_pruning = 0
    avg_reward_train_scratch = 0
    avg_reward_heuristic = 0
    closeness_env = closeness_environments[i]
    feature_env = feature_environments[i]
    combined_env = combined_environments[i]
    for j in range(diff_start_query_test):
        #set their starting query 
        start_query = new_queries[j]
        #x, y = 1, 2
        closeness_env.reset(new_query=start_query)
        feature_env.reset(new_query=start_query)
        combined_env.reset(new_query=start_query)
        print("Start Query: " + start_query)
        time_closeness, dag_closeness = train_q_qr(env=closeness_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_1_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
        time_feature, dag_feature = train_q_qr(env=feature_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_2_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
        time_combined, dag_combined = train_q_qr(env=combined_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_3_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
        inference_time_combined, reward_ground_truth, _ = inference_q_qr(env=combined_env, q_table_path=q_table_3_output_path, edge_dict=dag_combined.edge_dict)
        best_path, reward_exnonzero, total_time, pruning_percentage = run_pruning(env=combined_env, dag_1=dag_closeness, dag_2=dag_feature, discount_factor=discount_factor, learning_rate=learning_rate)
        max_allowed_path_size_heuristic = combined_env.state_count
        combined_env.reset()
        reward_heuristic, best_path_heuristic, paths_heuristic, shortest_paths_heuristic, total_time_heuristic = run_heuristic(q_table_1_path=q_table_1_output_path, q_table_2_path=q_table_2_output_path, k=heuristic_k, max_allowed_path_size=max_allowed_path_size_heuristic, dag_closeness=dag_closeness, dag_feature=dag_feature, env=combined_env)
        # reset agent position to try all paths and get rewards
        avg_reward_pruning += reward_exnonzero
        avg_reward_train_scratch += reward_ground_truth
        avg_reward_heuristic += reward_heuristic

    avg_reward_pruning = round(((avg_reward_pruning / diff_start_query_test)), 2)
    avg_reward_train_scratch = round(((avg_reward_train_scratch / diff_start_query_test)), 2)
    avg_reward_heuristic = round(((avg_reward_heuristic / diff_start_query_test)), 2)
    df.at[i, env_size_index] = int(combined_env.state_count)
    df.at[i, avg_cr_exnonzero_index] = avg_reward_pruning
    df.at[i, avg_cr_train_scratch_index] = avg_reward_train_scratch
    df.at[i, avg_cr_heuristic_index] = avg_reward_heuristic
    avg_rewards_exnonzero.append(avg_reward_pruning)
    avg_rewards_train_scratch.append(avg_reward_train_scratch)
    avg_rewards_heuristic.append(avg_reward_heuristic)
    df.to_csv(csv_file_name, index=False, header=header)


plot_cumulative_reward_qr(csv_file_name, header[0], header[1], header[2], header[3])
print("State space sizes: " + str(env_sizes))
print("Avg rewards for ExNonZeroDiscount: " + str(avg_rewards_exnonzero))
print("Avg rewards for Train from scratch: " + str(avg_rewards_train_scratch))
print("Avg rewards for Greedy-K: " + str(avg_rewards_heuristic))
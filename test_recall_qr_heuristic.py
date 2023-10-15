from env.init_query_refine import init_query_refine_2
from train_q_qr import train_q_qr
from inference_q_qr import inference_q_qr
import pandas as pd
from utilities import plot_recall_qr_heuristic
from heuristic_qr import run_heuristic


#inputs
env_test_count = 5
# 1 to 4
diff_start_query_test = 4
first_env_size = 7
env_test_step = 1
n_episodes = 2
k = 2
max_steps_per_episode = 4
agent_type = "QLearning"
_, new_queries = init_query_refine_2("closeness", first_env_size)

#output
recalls_heuristic = []
csv_file_name = "Test_Recall_QR_Greedy_" + str(k) + "_" + agent_type + ".csv"
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
header = ["State Space Size", "Recall - Greedy-K"]
env_size_index = 0
recall_heuristic_index = 1


for i in range(env_test_count):
    recall_heuristic = 0
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
        time_closeness, dag_closeness = train_q_qr(env=closeness_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_1_output_path)
        time_feature, dag_feature = train_q_qr(env=feature_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_2_output_path)
        time_combined, dag_combined = train_q_qr(env=combined_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_3_output_path)
        inference_time_combined, reward_ground_truth, _ = inference_q_qr(env=combined_env, q_table_path=q_table_3_output_path, edge_dict=dag_combined.edge_dict)
        combined_env.reset(new_query=start_query)
        max_reward, best_path, paths, shortest_paths, _ = run_heuristic(q_table_1_output_path, q_table_2_output_path, k, max_steps_per_episode, dag_closeness, dag_feature, env=combined_env)
        # reset agent position to try all paths and get rewards
        if max_reward >= reward_ground_truth:
            recall_heuristic += 1

    recall_heuristic = round(((recall_heuristic / diff_start_query_test) * 100), 2)
    df.at[i, env_size_index] = int(combined_env.state_count)
    df.at[i, recall_heuristic_index] = recall_heuristic
    recalls_heuristic.append(recall_heuristic)
    df.to_csv(csv_file_name, index=False, header=header)


plot_recall_qr_heuristic(csv_file_name, header[0], header[1], k)
print("State space sizes: " + str(env_sizes))
print("Recalls for Greedy-K with K = " + str(k) + ": " + str(recalls_heuristic))


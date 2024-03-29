from env.init_query_refine import init_query_refine_2
from train_q_qr import train_q_qr
from inference_q_qr import inference_q_qr
import pandas as pd
from utilities import plot_recalls_qr_closeness_feature


#inputs
env_test_count = 5
# 1 to 4
diff_start_query_test = 4
first_env_size = 7
env_test_step = 1
n_episodes = 3
max_steps_per_episode = 4
learning_rate = 0.1
discount_factor = 0.99
agent_type = "Sarsa"
_, new_queries = init_query_refine_2("closeness", first_env_size)

#output
recalls_closeness = []
recalls_feature = []
csv_file_name = "Test_recall_QR_close_feature_" + agent_type + ".csv"
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
header = ["State Space Size", "Recall - closeness", "Recall - feature"]
env_size_index = 0
recall_closeness_index = 1
recall_feature_index = 2


for i in range(env_test_count):
    recall_closeness = 0
    recall_feature = 0
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
        inference_time_combined, reward_closeness, _ = inference_q_qr(env=combined_env, q_table_path=q_table_1_output_path, edge_dict=dag_closeness.edge_dict)
        combined_env.reset(new_query=start_query)
        inference_time_combined, reward_feature, _ = inference_q_qr(env=combined_env, q_table_path=q_table_2_output_path, edge_dict=dag_feature.edge_dict)
        combined_env.reset(new_query=start_query)
        inference_time_combined, reward_ground_truth, _ = inference_q_qr(env=combined_env, q_table_path=q_table_3_output_path, edge_dict=dag_combined.edge_dict)
        # reset agent position to try all paths and get rewards
        if reward_closeness >= reward_ground_truth:
            recall_closeness += 1
        if reward_feature >= reward_ground_truth:
            reward_feature += 1

    recall_closeness = round(((recall_closeness / diff_start_query_test) * 100), 2)
    recall_feature = round(((recall_feature / diff_start_query_test) * 100), 2)
    df.at[i, env_size_index] = int(combined_env.state_count)
    df.at[i, recall_closeness_index] = recall_closeness
    df.at[i, recall_feature_index] = recall_feature
    recalls_closeness.append(recall_closeness)    
    recalls_feature.append(recall_feature)    
    df.to_csv(csv_file_name, index=False, header=header)


plot_recalls_qr_closeness_feature(csv_file_name, header[0], header[1], header[2])
print("State space sizes: " + str(env_sizes))
print("Recalls for Closeness: " + str(recalls_closeness))
print("Recalls for Feature: " + str(recalls_feature))


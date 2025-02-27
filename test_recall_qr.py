from env.init_query_refine import init_query_refine_2
from train_q_qr import train_q_qr
from inference_q_qr import inference_q_qr
from pruning_qr import run_pruning
import pandas as pd
from utilities import plot_recalls_qr


#inputs
env_test_count = 2
# 1 to 4
diff_start_query_test = 4
first_env_size = 7
env_test_step = 1
n_episodes = 2
max_steps_per_episode = 4
learning_rate = 0.1
discount_factor = 0.99
agent_type = "QLearning"
_, new_queries = init_query_refine_2("closeness", first_env_size)

#output
recalls_exact_pruning = []
csv_file_name = "Recall_QR_" + agent_type + ".csv"
q_table_1_output_path = "q_table_closeness.npy"
q_table_2_output_path = "q_table_feature.npy"
q_table_3_output_path = "q_table_combined.npy"
parameterized = True
alpha, beta = (2, 3)

env_sizes = []
for i in range(env_test_count):
    env_size = env_test_step * (i) + first_env_size
    env_sizes.append(env_size)

closeness_environments = []
feature_environments = []
combined_environments = []
for env_size in env_sizes:
    env_1, _ = init_query_refine_2("closeness", env_size, parameterized=parameterized, alpha_beta=(alpha, beta))
    env_2, _ = init_query_refine_2("feature", env_size, parameterized=parameterized, alpha_beta=(alpha, beta))
    env_3, _ = init_query_refine_2("combined", env_size, parameterized=parameterized, alpha_beta=(alpha, beta))
    closeness_environments.append(env_1)
    feature_environments.append(env_2)
    combined_environments.append(env_3)

# setup panda
df = pd.DataFrame()
header = ["State Space Size", "Recall - ExNonZeroDiscount"]
env_size_index = 0
recall_exact_pruning_index = 1


for i in range(env_test_count):
    recall_exact_pruning = 0
    recall_heuristic = 0
    recall_deep = 0
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
        time_closeness, dag_path = train_q_qr(env=closeness_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_1_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
        time_feature, dag_gold = train_q_qr(env=feature_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_2_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
        time_combined, dag_combined = train_q_qr(env=combined_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_3_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
        inference_time_combined, reward_ground_truth, _ = inference_q_qr(env=combined_env, q_table_path=q_table_3_output_path, edge_dict=dag_combined.edge_dict)
        best_path, max_reward, total_time, pruning_percentage = run_pruning(env=combined_env, dag_1=dag_path, dag_2=dag_gold, discount_factor=discount_factor, learning_rate=learning_rate)
        # reset agent position to try all paths and get rewards
        if max_reward >= reward_ground_truth:
            recall_exact_pruning += 1

    recall_exact_pruning = round(((recall_exact_pruning / diff_start_query_test) * 100), 2)
    df.at[i, env_size_index] = int(combined_env.state_count)
    df.at[i, recall_exact_pruning_index] = recall_exact_pruning
    recalls_exact_pruning.append(recall_exact_pruning)
    df.to_csv(csv_file_name, index=False, header=header)


plot_recalls_qr(csv_file_name, header[0], header[1])
print("State space sizes: " + str(env_sizes))
print("Recalls for ExNonZeroDiscount: " + str(recalls_exact_pruning))


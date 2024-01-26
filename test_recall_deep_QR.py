from env.init_query_refine import init_query_refine_2
import random
import pandas as pd
from utilities import plot_recall_deep
from train_deep_policy_qr import train_deep_qr
from Inference_deep_QR import inference_deep
from env.query_refine import Query_Refine

def get_random_start_query(new_queries):
    index = random.randint(0, len(new_queries) - 1)
    return new_queries[index]

def combine_paths(path_1, path_2):
    common_states = [state for state in path_1 if state in path_2]
    combined_paths = []
    for common_state in common_states:
        index_1 = path_1.index(common_state)
        index_2 = path_2.index(common_state)
        combined_path_1 = path_1[:index_1] + path_2[index_2:]
        combined_path_2 = path_2[:index_2] + path_1[index_1:]
        if combined_path_1 not in combined_paths:
            combined_paths.append(combined_path_1)
        if combined_path_2 not in combined_paths:
            combined_paths.append(combined_path_2)
    return combined_paths

def compute_deep_max_reward(env, paths):
    best_path = []
    max_reward = 0
    for path in paths:
        reward = 0
        for i in range(len(path) - 1):
            state_1 = path[i]
            state_2 = path[i + 1]
            action = Query_Refine.obtain_action(state_1, state_2, env.embed_size)
            _, r, done, _ = env.step(action)
            reward += r
        if reward > max_reward:
            max_reward = reward
            best_path = path
    return best_path, max_reward


#inputs
env_test_count = 5
diff_start_query_test = 4
first_env_size = 7
env_test_step = 1
deep_algorithm = "DQN"
timesteps = 2
_, new_queries = init_query_refine_2("closeness", first_env_size)

#output
recalls_deep = []
q_table_1_output_path = "q_table_closeness.npy"
q_table_2_output_path = "q_table_feature.npy"
q_table_3_output_path = "q_table_combined.npy"
deep_agent_closeness_policy_output_path = "agent_closeness.okl"
deep_agent_feature_policy_output_path = "agent_feature.okl"
deep_agent_combined_policy_output_path = "agent_combined.okl"

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
    closeness_environments.append(env_1[0])
    feature_environments.append(env_2[0])
    combined_environments.append(env_3[0])

# setup panda
df = pd.DataFrame()
header = ["Environment Size", "Recall - Deep Agent"]
env_size_index = 0
recall_deep_index = 1
csv_file_name = "Test_Recall_Deep_QR.csv"


for i in range(env_test_count):
    recall_deep = 0
    closeness_env = closeness_environments[i]
    feature_env = feature_environments[i]
    combined_env = combined_environments[i]
    for j in range(diff_start_query_test):
        query = get_random_start_query(new_queries)
        closeness_env.reset(query)
        feature_env.reset(query)
        combined_env.reset(query)
        print("Start Query: " + query)
        train_deep_qr(closeness_env, deep_algorithm, timesteps=timesteps, output_path=deep_agent_closeness_policy_output_path)
        train_deep_qr(feature_env, deep_algorithm, timesteps=timesteps, output_path=deep_agent_feature_policy_output_path)
        train_deep_qr(combined_env, deep_algorithm, timesteps=timesteps, output_path=deep_agent_combined_policy_output_path)
        path_1, _, _ = inference_deep(closeness_env, deep_algorithm, deep_agent_closeness_policy_output_path)
        path_2, _, _ = inference_deep(feature_env, deep_algorithm, deep_agent_feature_policy_output_path)
        path_3, reward_ground_truth, _ = inference_deep(combined_env, deep_algorithm, deep_agent_combined_policy_output_path)
        deep_all_paths = combine_paths(path_1, path_2)
        deep_best_path, deep_max_reward = compute_deep_max_reward(combined_env, deep_all_paths)
        if deep_max_reward >= reward_ground_truth:
            recall_deep += 1

    recall_deep = round(((recall_deep / diff_start_query_test) * 100), 2)
    df.at[i, env_size_index] = int(combined_env.state_count)
    df.at[i, recall_deep_index] = recall_deep
    recalls_deep.append(recall_deep)
    df.to_csv(csv_file_name, index=False, header=header)


plot_recall_deep(csv_file_name, header[0], header[1])
print("Environment sizes: " + str(env_sizes))
print("Recalls for Deep agent Query Refinement with deep algorithm = : " + deep_algorithm + ": " + str(recalls_deep))


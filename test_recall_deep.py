from env.init_gridworld import init_gridworld_3
import random
import pandas as pd
from utilities import plot_recall_deep
from train_deep_policy import train_deep
from Inference_deep import inference_deep

def get_random_start_pos(max_x, max_y):
    x = random.randint(0, max_x - 1)
    y = random.randint(0, max_y - 1)
    return x, y

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

def compute_deep_max_reward(gridworld, paths):
    best_path = []
    max_reward = 0
    for path in paths:
        reward = 0
        gridworld.reset()
        for i in range(len(path) - 1):
            state_1 = path[i]
            state_2 = path[i + 1]
            action = gridworld.obtain_action(state_1, state_2)
            _, r, done, _ = gridworld.step(action)
            reward += r
        if reward > max_reward:
            max_reward = reward
            best_path = path
    return best_path, max_reward


#inputs
env_test_count = 5
diff_agent_pos_per_test = 10
first_env_size = 4
env_test_step = 2
deep_algorithm = "DQN"
timesteps = 1000

#output
recalls_deep = []
q_table_1_output_path = "q_table_path.npy"
q_table_2_output_path = "q_table_gold.npy"
q_table_3_output_path = "q_table_combined.npy"
deep_agent_path_policy_output_path = "agent_path.okl"
deep_agent_gold_policy_output_path = "agent_gold.okl"
deep_agent_combined_policy_output_path = "agent_combined.okl"

env_sizes = []
for i in range(env_test_count):
    env_width = env_test_step * (i) + first_env_size
    env_length = env_width
    env_sizes.append((env_width, env_length))

path_environments = []
gold_environments = []
combined_environments = []
for (env_width, env_length) in env_sizes:
    grid_world_1 = init_gridworld_3("path", env_width, env_length)
    grid_world_2 = init_gridworld_3("gold", env_width, env_length)
    grid_world_3 = init_gridworld_3("combined", env_width, env_length)
    path_environments.append(grid_world_1)
    gold_environments.append(grid_world_2)
    combined_environments.append(grid_world_3)

# setup panda
df = pd.DataFrame()
header = ["Environment Size", "Recall - Deep Agent"]
env_size_index = 0
recall_deep_index = 1
csv_file_name = "Test_Recall_Deep.csv"


for i in range(env_test_count):
    recall_deep = 0
    path_env = path_environments[i]
    gold_env = gold_environments[i]
    combined_env = combined_environments[i]
    for j in range(diff_agent_pos_per_test):
        #set their starting position 
        x, y = get_random_start_pos(max_x=combined_env.target_position[0], max_y=combined_env.target_position[1])
        #x, y = 1, 2
        path_env.reset(new_start_position=[x, y])
        gold_env.reset(new_start_position=[x, y])
        combined_env.reset(new_start_position=[x, y])
        print("Start location: " + str((x, y)))
        train_deep(path_env, deep_algorithm, timesteps=timesteps, output_path=deep_agent_path_policy_output_path)
        train_deep(gold_env, deep_algorithm, timesteps=timesteps, output_path=deep_agent_gold_policy_output_path)
        train_deep(combined_env, deep_algorithm, timesteps=timesteps, output_path=deep_agent_combined_policy_output_path)
        path_1, cumulative_reward_deep_path, _ = inference_deep(path_env, deep_algorithm, deep_agent_path_policy_output_path)
        path_2, cumulative_reward_deep_gold, _ = inference_deep(gold_env, deep_algorithm, deep_agent_gold_policy_output_path)
        path_3, reward_ground_truth, _ = inference_deep(combined_env, deep_algorithm, deep_agent_combined_policy_output_path)
        deep_all_paths = combine_paths(path_1, path_2)
        deep_best_path, deep_max_reward = compute_deep_max_reward(combined_env, deep_all_paths)
        if deep_max_reward >= reward_ground_truth:
            recall_deep += 1

    recall_deep = round(((recall_deep / diff_agent_pos_per_test) * 100), 2)
    df.at[i, env_size_index] = int(combined_env.state_count)
    df.at[i, recall_deep_index] = recall_deep
    recalls_deep.append(recall_deep)
    df.to_csv(csv_file_name, index=False, header=header)


plot_recall_deep(csv_file_name, header[0], header[1])
print("Environment sizes: " + str(env_sizes))
print("Recalls for Deep agent with deep algorithm = : " + deep_algorithm + ": " + str(recalls_deep))


from env.init_gridworld import init_gridworld_3
from train_q_policy import train_q_policy
from inference_q import inference_q
import random
from pruning import run_pruning
import pandas as pd
from utilities import plot_recalls
from heuristic import run_heuristic
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
env_test_count = 2
diff_agent_pos_per_test = 1
first_env_size = 4
env_test_step = 1
n_episodes = 1000
max_steps_per_episode = 100
result_step_size = 10
learning_rate = 0.1
discount_factor = 0.99
heuristic_k = 1
agent_type = "QLearning"
deep_algorithm = "PPO"
timesteps = n_episodes

#output
recalls_exact_pruning = []
recalls_heuristic = []
recalls_deep = []
csv_file_name = "Recall_" + agent_type + "_" + ".csv"
q_table_1_output_path = "q_table_path.npy"
q_table_2_output_path = "q_table_gold.npy"
q_table_3_output_path = "q_table_combined.npy"
deep_agent_path_policy_output_path = "agent_path.okl"
deep_agent_gold_policy_output_path = "agent_gold.okl"

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
header = ["Environment Size", "Recall - ExNonZeroDiscount", "Recall - Greedy K", "Recall - Deep Agent"]
env_size_index = 0
recall_exact_pruning_index = 1
recall_heuristic_index = 2
recall_deep_index = 3


for i in range(env_test_count):
    recall_exact_pruning = 0
    recall_heuristic = 0
    recall_deep = 0
    path_env = path_environments[i]
    gold_env = gold_environments[i]
    combined_env = combined_environments[i]
    heuristic_max_allowed_path_size = combined_env.grid_width + combined_env.grid_length
    for j in range(diff_agent_pos_per_test):
        #set their starting position 
        x, y = get_random_start_pos(max_x=combined_env.target_position[0], max_y=combined_env.target_position[1])
        #x, y = 1, 2
        path_env.reset(new_start_position=[x, y])
        gold_env.reset(new_start_position=[x, y])
        combined_env.reset(new_start_position=[x, y])
        print("Start location: " + str((x, y)))
        time_path, dag_path, _ = train_q_policy(grid_world=path_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_1_output_path, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
        time_gold, dag_gold, _ = train_q_policy(grid_world=gold_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_2_output_path, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
        time_combined, dag_combined, _ = train_q_policy(grid_world=combined_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_3_output_path, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
        inference_time_combined, reward_ground_truth, _ = inference_q(grid_world=combined_env, q_table_path=q_table_3_output_path)
        best_path, max_reward, total_time, pruning_percentage = run_pruning(gridworld=combined_env, dag_1=dag_path, dag_2=dag_gold, discount_factor=discount_factor, learning_rate=learning_rate)
        # reset agent position to try all paths and get rewards
        if max_reward >= reward_ground_truth:
            recall_exact_pruning += 1
        combined_env.reset(new_start_position=[x, y])
        max_cumulative_reward, best_path, paths, shortest_paths, total_time = run_heuristic(q_table_1_output_path, q_table_2_output_path, heuristic_k, heuristic_max_allowed_path_size, gridworld=combined_env)
        if max_cumulative_reward >= reward_ground_truth:
            recall_heuristic += 1
        path_env.reset(new_start_position=[x, y])
        gold_env.reset(new_start_position=[x, y])
        train_deep(path_env, deep_algorithm, timesteps=timesteps, output_path=deep_agent_path_policy_output_path)
        train_deep(gold_env, deep_algorithm, timesteps=timesteps, output_path=deep_agent_gold_policy_output_path)
        path_1, cumulative_reward_deep_path, _ = inference_deep(path_env, deep_algorithm, deep_agent_path_policy_output_path)
        path_2, cumulative_reward_deep_gold, _ = inference_deep(gold_env, deep_algorithm, deep_agent_gold_policy_output_path)
        deep_all_paths = combine_paths(path_1, path_2)
        deep_best_path, deep_max_reward = compute_deep_max_reward(combined_env, deep_all_paths)
        if deep_max_reward >= reward_ground_truth:
            recall_deep += 1

    recall_exact_pruning = round(((recall_exact_pruning / diff_agent_pos_per_test) * 100), 2)
    recall_heuristic = round(((recall_heuristic / diff_agent_pos_per_test) * 100), 2)
    recall_deep = round(((recall_deep / diff_agent_pos_per_test) * 100), 2)
    df.at[i, env_size_index] = str((combined_env.grid_width, combined_env.grid_length))
    df.at[i, recall_exact_pruning_index] = recall_exact_pruning
    df.at[i, recall_heuristic_index] = recall_heuristic
    df.at[i, recall_deep_index] = recall_deep
    recalls_exact_pruning.append(recall_exact_pruning)
    recalls_heuristic.append(recall_heuristic)
    recalls_deep.append(recall_deep)
    df.to_csv(csv_file_name, index=False, header=header)


plot_recalls(csv_file_name, header[0], header[1], header[2], header[3])
print("Environment sizes: " + str(env_sizes))
print("Recalls for ExNonZeroDiscount: " + str(recalls_exact_pruning))
print("Recalls for greedy-k algorithm with k = : " + str(heuristic_k) + ": " + str(recalls_heuristic))
print("Recalls for Deep agent with deep algorithm = : " + deep_algorithm + ": " + str(recalls_deep))


from env.init_gridworld import init_gridworld_3
from train_q_policy import train_q_policy
from inference_q import inference_q
import random
from pruning import run_pruning
import pandas as pd
from utilities import plot_recalls

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

#inputs
env_test_count = 3
diff_agent_pos_per_test = 1
first_env_size = 4
env_test_step = 1
n_episodes = 1000
max_steps_per_episode = 100
result_step_size = 10
learning_rate = 0.1
discount_factor = 0.99
agent_type = "QLearning"
# A2C
parameterized = True
timesteps = n_episodes

#output
recalls_exact_pruning = []
csv_file_name = "Recall_" + agent_type + "_" + ".csv"
q_table_1_output_path = "q_table_path.npy"
q_table_2_output_path = "q_table_gold.npy"
q_table_3_output_path = "q_table_combined.npy"

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
    grid_world_3 = init_gridworld_3("combined", env_width, env_length, parameterized=parameterized)
    path_environments.append(grid_world_1)
    gold_environments.append(grid_world_2)
    combined_environments.append(grid_world_3)

# setup panda
df = pd.DataFrame()
header = ["Environment Size", "Recall - ExNonZeroDiscount"]
env_size_index = 0
recall_exact_pruning_index = 1


for i in range(env_test_count):
    recall_exact_pruning = 0
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
        time_path, dag_path, _, _ = train_q_policy(grid_world=path_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_1_output_path, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
        time_gold, dag_gold, _, _ = train_q_policy(grid_world=gold_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_2_output_path, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
        time_combined, dag_combined, _, _ = train_q_policy(grid_world=combined_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_3_output_path, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
        inference_time_combined, reward_ground_truth, _ = inference_q(grid_world=combined_env, q_table_path=q_table_3_output_path)
        best_path, max_reward, total_time, pruning_percentage = run_pruning(gridworld=combined_env, dag_1=dag_path, dag_2=dag_gold, discount_factor=discount_factor, learning_rate=learning_rate)
        # reset agent position to try all paths and get rewards
        if max_reward >= reward_ground_truth:
            recall_exact_pruning += 1
        combined_env.reset(new_start_position=[x, y])
        path_env.reset(new_start_position=[x, y])
        gold_env.reset(new_start_position=[x, y])

    recall_exact_pruning = round(((recall_exact_pruning / diff_agent_pos_per_test) * 100), 2)
    df.at[i, env_size_index] = str((combined_env.grid_width, combined_env.grid_length))
    df.at[i, recall_exact_pruning_index] = recall_exact_pruning
    recalls_exact_pruning.append(recall_exact_pruning)
    df.to_csv(csv_file_name, index=False, header=header)


plot_recalls(csv_file_name, header[0], header[1])
print("Environment sizes: " + str(env_sizes))
print("Recalls for ExNonZeroDiscount: " + str(recalls_exact_pruning))


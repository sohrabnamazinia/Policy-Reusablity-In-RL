from env.init_gridworld import init_gridworld_3
from train_q_policy import train_q_policy
from inference_q import inference_q
import random
import pandas as pd


def get_random_start_pos(max_x, max_y):
    x = random.randint(0, max_x - 1)
    y = random.randint(0, max_y - 1)
    return x, y


#inputs
env_test_count = 5
diff_agent_pos_per_test = 5
first_env_size = 4
env_test_step = 2
n_episodes = 1000
max_steps_per_episode = 100
result_step_size = 10
learning_rate = 0.1
discount_factor = 0.99
agent_type = "QLearning"


#output
recalls_path = []
recalls_gold = []

csv_file_name = "Baseline_" + agent_type + ".csv"
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
    grid_world_3 = init_gridworld_3("combined", env_width, env_length)
    path_environments.append(grid_world_1)
    gold_environments.append(grid_world_2)
    combined_environments.append(grid_world_3)

# setup panda
df = pd.DataFrame()
header = ["Environment Size", "Recall - path", "Recall - gold"]
env_size_index = 0
recall_path_index = 1
recall_gold_index = 2


for i in range(env_test_count):
    recall_path = 0
    recall_gold = 0

    path_env = path_environments[i]
    gold_env = gold_environments[i]
    combined_env = combined_environments[i]

    for j in range(diff_agent_pos_per_test):
        #set their starting position 
        x, y = get_random_start_pos(max_x=combined_env.target_position[0], max_y=combined_env.target_position[1])

        path_env.reset(new_start_position=[x, y])
        gold_env.reset(new_start_position=[x, y])
        combined_env.reset(new_start_position=[x, y])

        time_path, dag_path, _ = train_q_policy(grid_world=path_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_1_output_path, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
        time_gold, dag_gold, _ = train_q_policy(grid_world=gold_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_2_output_path, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
        time_combined, dag_combined, _ = train_q_policy(grid_world=combined_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_3_output_path, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
        inference_time_path, reward_path, _ = inference_q(grid_world=combined_env, q_table_path=q_table_1_output_path)
        inference_time_gold, reward_gold, _ = inference_q(grid_world=combined_env, q_table_path=q_table_2_output_path)
        inference_time_combined, reward_ground_truth, _ = inference_q(grid_world=combined_env, q_table_path=q_table_3_output_path)

        # reset agent position to try all paths and get rewards
        if reward_path >= reward_ground_truth:
            recall_path += 1
        if reward_gold >= reward_ground_truth:
            recall_gold += 1

        combined_env.reset(new_start_position=[x, y])
        path_env.reset(new_start_position=[x, y])
        gold_env.reset(new_start_position=[x, y])


    recall_path = round(((recall_path / diff_agent_pos_per_test) * 100), 2)
    recall_gold = round(((recall_gold / diff_agent_pos_per_test) * 100), 2)

    df.at[i, env_size_index] = str((combined_env.grid_width, combined_env.grid_length))
    df.at[i, recall_path_index] = recall_path
    df.at[i, recall_gold_index] = recall_gold

    recalls_path.append(recall_path)
    recalls_gold.append(recall_gold)

    df.to_csv(csv_file_name, index=False, header=header)
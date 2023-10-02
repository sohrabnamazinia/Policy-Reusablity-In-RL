from env.init_gridworld import init_gridworld_4
from train_q_policy import train_q_policy
import pandas as pd
from utilities import plot_cumulative_reward_env_size
from inference_q import inference_q
from pruning import run_pruning
from heuristic import run_heuristic

#inputs
env_test_count = 3
first_env_size = 4
env_test_step = 1
n_episodes = 1000
max_steps_per_episode = 100
result_step_size = 10
learning_rate = 0.1
discount_factor = 0.99
heuristic_k = 2
reward_system = "combined"
agent_type = "Sarsa"

#output
cumulative_rewards_train_combined = []
cumulative_rewards_pruning = []
cumulative_rewards_greedy_k = []
csv_file_name = "cumulative_reward_" + agent_type + "_" + str(n_episodes) + ".csv"
q_table_output_path_1 = "q_table_path.npy" 
q_table_output_path_2 = "q_table_gold.npy" 
q_table_output_path_3 = "q_table_combined.npy" 

# setup plot
header = ["Environment Size", "Cumulative Reward - Train Combined", "Cumulative Reward - ExNonZeroDiscount", "Cumulative Reward - Greedy K"]
environment_size_index = 0
cumulative_reward_Train_Combined_index = 1
cumulative_reward_ExNonZeroDiscount_index = 2
cumulative_reward_greedy_k_index = 3
df = pd.DataFrame()

env_sizes = []
for i in range(env_test_count):
    env_width = env_test_step * (i) + first_env_size
    env_length = env_width
    env_sizes.append((env_width, env_length))


path_environments = []
gold_environments = []
combined_environments = []
for (env_width, env_length) in env_sizes:
    grid_world = init_gridworld_4(reward_system, env_width, env_length)
    path_environments.append(grid_world)
    gold_environments.append(grid_world)
    combined_environments.append(grid_world)

for i in range(env_test_count):
    path_env = path_environments[i]
    gold_env = gold_environments[i]
    combined_env = combined_environments[i]
    total_time_1, dag_1, _ = train_q_policy(path_env, n_episodes, max_steps_per_episode, agent_type, q_table_output_path_1, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
    total_time_2, dag_2, _ = train_q_policy(gold_env, n_episodes, max_steps_per_episode, agent_type, q_table_output_path_2, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
    total_time_3, dag_3, _ = train_q_policy(combined_env, n_episodes, max_steps_per_episode, agent_type, q_table_output_path_3, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
    _, cumulative_reward_train_combined, path = inference_q(grid_world=combined_env, q_table_path=q_table_output_path_1)
    best_path, cumulative_reward_pruning, total_time, pruning_percentage = run_pruning(combined_env, dag_1=dag_1, dag_2=dag_2, discount_factor=discount_factor, learning_rate=learning_rate)
    cumulative_reward_greedy_k, best_path_greedy_k, _, _, _ = run_heuristic(q_table_output_path_1, q_table_output_path_2, heuristic_k, max_steps_per_episode, gridworld=combined_env)
    
    cumulative_rewards_train_combined.append(cumulative_reward_train_combined)
    cumulative_rewards_pruning.append(cumulative_reward_pruning)
    cumulative_rewards_greedy_k.append(cumulative_reward_greedy_k)
    df.at[i, environment_size_index] = str((combined_env.grid_width, combined_env.grid_length))
    df.at[i, cumulative_reward_Train_Combined_index] = cumulative_reward_train_combined
    df.at[i, cumulative_reward_ExNonZeroDiscount_index] = cumulative_reward_pruning
    df.at[i, cumulative_reward_greedy_k_index] = cumulative_reward_greedy_k
    df.to_csv(csv_file_name, index=False, header=header)

print("Cumulative rewards for train combined:\n" + str(cumulative_rewards_train_combined))
print("Cumulative rewards for ExNonZeroDiscount:\n" + str(cumulative_rewards_pruning))
print("Cumulative rewards for Greedy K:\n" + str(cumulative_rewards_greedy_k))
plot_cumulative_reward_env_size(csv_file_name, header[0], header[1], header[2], header[3])
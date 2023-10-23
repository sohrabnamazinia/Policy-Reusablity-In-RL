from env.init_gridworld import init_gridworld_6
import pandas as pd
from train_q_policy import train_q_policy
from inference_q import inference_q
from pruning import run_pruning
from heuristic import run_heuristic
from utilities import plot_speedup_action_size

#inputs
k_test_count = 5
first_k = 1
k_test_step = 1
n_episodes = 1000
max_steps_per_episode = 100
env_side_length = 5
agent_type = "QLearning"

#output
times_greedy_k = []
Cumulative_rewards_greedy_k = []
csv_file_name = "Test_greedy_k_tradeoff_" + agent_type + ".csv"

q_table_1_output_path = "q_table_1.npy"
q_table_2_output_path = "q_table_2.npy"
q_table_3_output_path = "q_table_combined.npy"

Ks = []
for i in range(k_test_count):
    k = k_test_step * (i) + first_k
    Ks.append(k)

policy_1_environments = []
policy_2_environments = []
for action_size in Ks:
    grid_world_1 = init_gridworld_6("path", action_size=action_size, side_length=env_side_length)
    grid_world_2 = init_gridworld_6("gold", action_size=action_size, side_length=env_side_length)
    policy_1_environments.append(grid_world_1)
    policy_2_environments.append(grid_world_2)

# setup panda
df = pd.DataFrame()
header = ["K", "Time - Greedy-K", "Cumulative Reward - Greedy-k"]
k_index = 0
time_greedy_k_index = 1
Cumulative_reward_greedy_k_index = 2

for i in range(k_test_count):
    k = Ks[i]
    path_env = policy_1_environments[i]
    gold_env = policy_2_environments[i]
    time_path, dag_path, _ = train_q_policy(grid_world=path_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_1_output_path)
    time_gold, dag_gold, _ = train_q_policy(grid_world=gold_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_2_output_path)
    Cumulative_reward_greedy_k, _, _, _, time_greedy_k = run_heuristic(q_table_1_output_path, q_table_2_output_path, k=k, max_allowed_path_size=(2*env_side_length)-2, gridworld=path_env)

    df.at[i, k_index] = k
    df.at[i, time_greedy_k_index] = time_greedy_k
    df.at[i, Cumulative_reward_greedy_k_index] = Cumulative_reward_greedy_k
    times_greedy_k.append(time_greedy_k)
    Cumulative_rewards_greedy_k.append(Cumulative_reward_greedy_k)
    df.to_csv(csv_file_name, index=False, header=header)

print("Action sizes: " + str(Ks))
print("Total times for greedy-k algorithm: " + str(times_greedy_k))
print("Total rewards for greedy-k algorithm: " + str(Cumulative_rewards_greedy_k))
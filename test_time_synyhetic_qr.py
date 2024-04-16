from env.init_query_refine import init_query_refine_2
import pandas as pd
from train_q_qr import train_q_qr
from inference_q_qr import inference_q_qr
from pruning_synthetic_QR import run_pruning_qr
from utilities import plot_times
from heuristic_synthetic_QR import run_heuristic

#inputs
env_size = 7
max_steps_per_episode = 4
agent_type = "QLearning"
n_episodes = 4
max_steps_per_episode = 100
result_step_size = 10
learning_rate = 0.1
discount_factor = 0.99
agent_type = "QLearning"
num_synthetic_policies = 12

synthetic_reward = [i for i in range(4, num_synthetic_policies+1, 2)]

#output
times_train_scratch = []
times_ExNonZeroDiscount = []
#times_greedy_k = []
csv_file_name = "Test_Time_" + agent_type + "_synthetic_QR.csv"

q_tables = {}


for i in range(num_synthetic_policies):
    q_tables[f"q_table_{i}_output_path"] = f"q_table_R{i}_QR.npy"

q_table_combined_output_path = "q_table_combined_QR.npy"

def create_policy_environments(num_synthetic_policies):
    policy_environments = {}
    for i in range(num_synthetic_policies):
        variable_name = f"policy_R{i}_environments"  # Construct the variable name
        policy_environments[variable_name] = []  # Create an empty list with the variable name
    return policy_environments

policy_environments = create_policy_environments(num_synthetic_policies)
combined_environments = []

def create_env(num_synthetic_policies):
    envs = {}
    for i in range(num_synthetic_policies):
        variable_name = f"env_{i}"  # Construct the variable name
        envs[variable_name] = None
    return envs

envs = create_env(num_synthetic_policies)



for i in range(num_synthetic_policies):
    envs[f"env_{i}"] = init_query_refine_2(f"R{i}", env_size)
    policy_environments[f"policy_R{i}_environments"].append(envs[f"env_{i}"])
grid_world_combined = init_query_refine_2("combined_synthetic", env_size)
combined_environments.append(grid_world_combined)

for key in policy_environments.keys():
    print(key, len(policy_environments[key]))
# setup panda
df = pd.DataFrame()
#header = ["Number of Synthetic Rewards", "Time - ExNonZeroDiscount", "Time - train scratch", "Time - Greedy_K"]
header = ["Number of Synthetic Rewards", "Time - ExNonZeroDiscount", "Time - train scratch"]
reward_index = 0
time_ExNonZeroDiscount_index = 1
time_train_scratch_index = 2
#greedy_k_index = 3

synthetic_env = {}
lstDAG = []

for i, n in enumerate(synthetic_reward):
    for j in range(n):
        synthetic_env[f"R{j}_env"] = policy_environments[f"policy_R{j}_environments"][0]
        time, dag = train_q_qr(env=synthetic_env[f"R{j}_env"][0], n_episodes=n_episodes,
                                                max_steps_per_episode=max_steps_per_episode, agent_type=agent_type,
                                                output_path=q_tables[f"q_table_{j}_output_path"],
                                                learning_rate=learning_rate, discount_factor=discount_factor)
        lstDAG.append(dag)

    combined_env = combined_environments[0]
    #heuristic_max_allowed_path_size = combined_env.grid_width + combined_env.grid_length

    time_train_combined, dag_combined = train_q_qr(env=combined_env[0], n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_combined_output_path,learning_rate=learning_rate, discount_factor=discount_factor)
    inference_time_combined, reward_ground_truth, _ = inference_q_qr(env=combined_env[0], q_table_path=q_table_combined_output_path, edge_dict=dag_combined.edge_dict)
    time_from_scratch = time_train_combined + inference_time_combined
    best_path, max_reward, time_ExNonZeroDiscount, pruning_percentage = run_pruning_qr(env=combined_env[0], dags=lstDAG, discount_factor=discount_factor, learning_rate=learning_rate, number_of_episodes=n_episodes)

    #max_cumulative_reward, best_path, paths, shortest_paths, time_greedy_k = run_heuristic(q_tables_path=q_tables, k=heuristic_k, n=n, max_allowed_path_size=heuristic_max_allowed_path_size, gridworld=combined_env)
    df.at[i, reward_index] = n
    df.at[i, time_ExNonZeroDiscount_index] = time_ExNonZeroDiscount
    df.at[i, time_train_scratch_index] = time_from_scratch
    #df.at[i, greedy_k_index] = time_greedy_k
    times_train_scratch.append(time_from_scratch)
    times_ExNonZeroDiscount.append(time_ExNonZeroDiscount)
    #times_greedy_k.append(time_greedy_k)
    df.to_csv(csv_file_name, index=False, header=header)

#plot_times(csv_file_name, header[0], header[1], header[2], header[3])
plot_times(csv_file_name, header[0], header[1], header[2])
print("Number of Synthetic Rewards: " + str(synthetic_reward))
print("Total time for ExNonZeroDiscount: " + str(times_ExNonZeroDiscount))
print("Total time for train from scratch algorithm: " + str(times_train_scratch))
#print("Total time for Greedy-K: " + str(times_greedy_k))
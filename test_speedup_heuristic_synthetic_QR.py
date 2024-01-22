from env.init_query_refine import init_query_refine_2
import pandas as pd
from train_q_qr import train_q_qr
from inference_q_qr import inference_q_qr
from heuristic_synthetic_QR import run_heuristic
from utilities import plot_speedup

#inputs
n_episodes = 2
max_steps_per_episode = 4
heuristic_k = 2
learning_rate = 0.1
discount_factor = 0.99
agent_type = "Sarsa"
embedding_size = 7
# num_synthetic_policies = 12
num_synthetic_policies = 12

synthetic_rewards_Counts = [i for i in range(4, num_synthetic_policies+1, 2)]

#output
times_train_scratch = []
times_heuristic = []
speedups = []
csv_file_name = "Test_Speedup_" + agent_type + "_heuristic_synthetic_QR.csv"

q_tables = {}


for i in range(num_synthetic_policies):
    q_tables[f"q_table_{i}_output_path"] = f"q_table_R{i}.npy"

q_table_combined_output_path = "q_table_combined.npy"

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
    envs[f"env_{i}"] = init_query_refine_2(f"R{i}", embedding_size)
    policy_environments[f"policy_R{i}_environments"].append(envs[f"env_{i}"])
env_combined = init_query_refine_2("combined_synthetic", embedding_size)
combined_environments.append(env_combined)

for key in policy_environments.keys():
    print(key, len(policy_environments[key]))
# setup panda
df = pd.DataFrame()
header = ["Number of Synthetic Rewards", "Time - Greedy-K", "Time - train scratch", "Speedup"]
reward_index = 0
time_heuristic_index = 1
time_train_scratch_index = 2
speedup_index = 3

synthetic_env = {}
lstDAG = []

for i, n in enumerate(synthetic_rewards_Counts):
    for j in range(n):
        synthetic_env[f"R{j}_env"] = policy_environments[f"policy_R{j}_environments"][0]
        time, dag = train_q_qr(env=synthetic_env[f"R{j}_env"][0], n_episodes=n_episodes,
                                                max_steps_per_episode=max_steps_per_episode, agent_type=agent_type,
                                                output_path=q_tables[f"q_table_{j}_output_path"],
                                                learning_rate=learning_rate, discount_factor=discount_factor)
        lstDAG.append(dag)

    combined_env = combined_environments[0][0]

    time_train_combined, dag_combined = train_q_qr(env=combined_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_combined_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
    inference_time_combined, reward_ground_truth, _ = inference_q_qr(env=combined_env, q_table_path=q_table_combined_output_path, edge_dict=dag_combined.edge_dict)
    time_from_scratch = time_train_combined + inference_time_combined
    heuristic_max_allowed_path_size = combined_env.state_count
    combined_env.reset()
    max_cumulative_reward, best_path, paths, shortest_paths, time_heuristic = run_heuristic(dags=lstDAG, k=heuristic_k, n=n, max_allowed_path_size=heuristic_max_allowed_path_size, env=combined_env)

    speedup = time_from_scratch / time_heuristic
    df.at[i, reward_index] = n
    df.at[i, time_heuristic_index] = time_heuristic
    df.at[i, time_train_scratch_index] = time_from_scratch
    df.at[i, speedup_index] = speedup
    times_train_scratch.append(time_from_scratch)
    times_heuristic.append(time_heuristic)
    speedups.append(speedup)
    df.to_csv(csv_file_name, index=False, header=header)

plot_speedup(csv_file_name, header[0], header[1], header[2], header[3])
print("Number of Synthetic Rewards: " + str(synthetic_rewards_Counts))
print("Total time for Heuristic: " + str(times_heuristic))
print("Total time for train-from-scratch algorithm: " + str(times_train_scratch))
print("Speedups: " + str(speedups))
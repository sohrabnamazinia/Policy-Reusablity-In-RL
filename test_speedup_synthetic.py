from env.init_gridworld import init_gridworld_3
import pandas as pd
from train_q_policy import train_q_policy
from inference_q import inference_q
from pruning_synthetic import run_pruning
from utilities import plot_speedup

#inputs
n_episodes = 1000
max_steps_per_episode = 100
result_step_size = 10
learning_rate = 0.1
discount_factor = 0.99
agent_type = "Sarsa"
env_width_size = 8
env_length_size = 8
num_synthetic_policies = 12

synthetic_reward = [i for i in range(4, num_synthetic_policies+1, 2)]

#output
times_train_scratch = []
times_ExNonZeroDiscount = []
speedups = []
csv_file_name = "Test_Speedup_" + agent_type + "_synthetic_correct_2nd.csv"

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

def create_grid_world(num_synthetic_policies):
    grid_worlds = {}
    for i in range(num_synthetic_policies):
        variable_name = f"grid_world_{i}"  # Construct the variable name
        grid_worlds[variable_name] = None
    return grid_worlds

grid_worlds = create_grid_world(num_synthetic_policies)



for i in range(num_synthetic_policies):
    grid_worlds[f"grid_world_{i}"] = init_gridworld_3(f"R{i}", env_width_size, env_length_size)
    policy_environments[f"policy_R{i}_environments"].append(grid_worlds[f"grid_world_{i}"])
grid_world_combined = init_gridworld_3("combined_synthetic", env_width_size, env_length_size)
combined_environments.append(grid_world_combined)

for key in policy_environments.keys():
    print(key, len(policy_environments[key]))
# setup panda
df = pd.DataFrame()
header = ["Number of Synthetic Rewards", "Time - ExNonZeroDiscount", "Time - train scratch", "Speedup"]
reward_index = 0
time_ExNonZeroDiscount_index = 1
time_train_scratch_index = 2
speedup_index = 3

synthetic_env = {}
lstDAG = []

for i, n in enumerate(synthetic_reward):
    for j in range(n):
        synthetic_env[f"R{j}_env"] = policy_environments[f"policy_R{j}_environments"][0]
        time, dag, _ = train_q_policy(grid_world=synthetic_env[f"R{j}_env"], n_episodes=n_episodes,
                                                max_steps_per_episode=max_steps_per_episode, agent_type=agent_type,
                                                output_path=q_tables[f"q_table_{j}_output_path"], result_step_size=result_step_size,
                                                learning_rate=learning_rate, discount_factor=discount_factor)
        lstDAG.append(dag)

    combined_env = combined_environments[0]

    time_train_combined, dag_combined, _ = train_q_policy(grid_world=combined_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_combined_output_path, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
    inference_time_combined, reward_ground_truth, _ = inference_q(grid_world=combined_env, q_table_path=q_table_combined_output_path)
    time_from_scratch = time_train_combined + inference_time_combined
    best_path, max_reward, time_ExNonZeroDiscount, pruning_percentage = run_pruning(gridworld=combined_env, dags=lstDAG, discount_factor=discount_factor, learning_rate=learning_rate, number_of_episodes=n_episodes)

    speedup = time_from_scratch / time_ExNonZeroDiscount
    df.at[i, reward_index] = n
    df.at[i, time_ExNonZeroDiscount_index] = time_ExNonZeroDiscount
    df.at[i, time_train_scratch_index] = time_from_scratch
    df.at[i, speedup_index] = speedup
    times_train_scratch.append(time_from_scratch)
    times_ExNonZeroDiscount.append(time_ExNonZeroDiscount)
    speedups.append(speedup)
    df.to_csv(csv_file_name, index=False, header=header)

plot_speedup(csv_file_name, header[0], header[1], header[2], header[3])
print("Number of Synthetic Rewards: " + str(synthetic_reward))
print("Total time for ExNonZeroDiscount: " + str(times_ExNonZeroDiscount))
print("Total time for greedy-k algorithm: " + str(times_train_scratch))
print("Speedups: " + str(speedups))
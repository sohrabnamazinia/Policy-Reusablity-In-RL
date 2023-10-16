from env.init_gridworld import init_gridworld_5
from train_q_policy import train_q_policy
from pruning_synthetic import run_pruning
from utilities import plot_discount_factors
import pandas as pd

def set_discount_factors(n):
    discount_factors = [0]
    for i in range(1, n):
        discount_factors.append(discount_factors[i - 1] + (1 / experiments_count))
    return discount_factors

# This variable is the input of this file
experiments_count = 10
n_episodes = 1000
max_steps_per_episode = 100
learning_rate = 0.1
env_width_size = 6
env_length_size = 6
agent_type = "QLearning"
num_synthetic_policies = 10

# outputs

output_paths = {}
policies = {}

for i in range(num_synthetic_policies):
    output_paths[f"output_path_{i}"] = f"q_table_R{i}.npy"
    policies[f"policy_{i}"] = f"R{i}"

output_path_combined = "q_table_combined.npy"
policy_combined = "combined"

csv_file_name = "Test_discount_factor_" + agent_type + "_synthetic.csv"

# setup plot
header = ["Discount Factor", "Pruning Percentage (%)"]
discount_factor_index = 0
pruning_percentage_index = 1
data_frame = pd.DataFrame()

# This variable is the output of this file
pruning_percentages = []
discount_factors = set_discount_factors(experiments_count)
gridWorlds = {}
lstDAG = []

for i in range(experiments_count):
    if (i == 0):
        pruning_percentages.append(100)
        continue
    df = discount_factors[i]
    for j in range(num_synthetic_policies):
        gridWorlds[f"gridworld_{j}"] = init_gridworld_5(policies[f"policy_{j}"], env_width_size, env_length_size)
        _, dag, _ = train_q_policy(gridWorlds[f"gridworld_{j}"], n_episodes, max_steps_per_episode, agent_type,  output_paths[f"output_path_{j}"],
                                     discount_factor=df, learning_rate=learning_rate)
        lstDAG.append(dag)

    gridworld_combined = init_gridworld_5(policy_combined, env_width_size, env_length_size)
    best_path, max_reward, total_time, pruning_percentage = run_pruning(gridworld_combined, lstDAG, learning_rate, df)
    pruning_percentages.append(pruning_percentage)
    data_frame.at[i, discount_factor_index] = df
    data_frame.at[i, pruning_percentage_index] = pruning_percentage
    data_frame.to_csv(csv_file_name, index=False, header=header)


plot_discount_factors(discount_factors, pruning_percentages)
print("Experiment results:\n")
print("Discount factors: " + str(discount_factors))
print("Pruning Percentages: " + str(pruning_percentages))
print("*****************************")


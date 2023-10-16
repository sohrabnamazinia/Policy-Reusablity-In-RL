from env.init_gridworld import init_gridworld_3
from train_q_policy import train_q_policy
from inference_q import inference_q
import random
from pruning_synthetic import run_pruning
import pandas as pd
# import psutil
import tracemalloc


def memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)  # Convert to megabytes


#inputs
n_episodes = 1000
max_steps_per_episode = 100
result_step_size = 10
learning_rate = 0.1
discount_factor = 0.99
agent_type = "QLearning"
env_width_size = 8
env_length_size = 8
num_synthetic_policies = 12

synthetic_reward = [i for i in range(4, num_synthetic_policies+1, 2)]

#output
memories_exact = []
memories_train_scratch = []

csv_file_name = "Memory_" + agent_type + "_2nd.csv"


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

# setup panda
df = pd.DataFrame()
header = ["Number of Synthetic Rewards", "Memory - ExNonZeroDiscount", "Memory - Train from Scratch",  "Memory - Individual Tasks"]
reward_index = 0
memory_exact_index_peak = 1
memory_train_scratch_index_peak = 2
memory_tasks_index_peak = 3


synthetic_env = {}
lstDAG = []



for i, n in enumerate(synthetic_reward):

    tracemalloc.start()
    memory_task = 0
    memory_train = 0
    for j in range(n):
        synthetic_env[f"R{j}_env"] = policy_environments[f"policy_R{j}_environments"][0]
        time, dag, _ = train_q_policy(grid_world=synthetic_env[f"R{j}_env"], n_episodes=n_episodes,
                                      max_steps_per_episode=max_steps_per_episode, agent_type=agent_type,
                                      output_path=q_tables[f"q_table_{j}_output_path"],
                                      result_step_size=result_step_size,
                                      learning_rate=learning_rate, discount_factor=discount_factor)
        lstDAG.append(dag)
        current_train, peak_train = tracemalloc.get_traced_memory()
        memory_task += peak_train
    tracemalloc.stop()

    combined_env = combined_environments[0]

    tracemalloc.start()
    time_combined, dag_combined, _ = train_q_policy(grid_world=combined_env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_combined_output_path, result_step_size=result_step_size, learning_rate=learning_rate, discount_factor=discount_factor)
    current_scratch_train, peak_scratch_train = tracemalloc.get_traced_memory()
    memory_train += peak_scratch_train

    inference_time_combined, reward_ground_truth, _ = inference_q(grid_world=combined_env, q_table_path=q_table_combined_output_path)

    current_scratch_infer, peak_scratch_infer = tracemalloc.get_traced_memory()
    memory_train += peak_scratch_infer

    tracemalloc.stop()

    tracemalloc.start()

    best_path, max_reward, total_time, pruning_percentage = run_pruning(gridworld=combined_env, dags=lstDAG, discount_factor=discount_factor, learning_rate=learning_rate, number_of_episodes=n_episodes)

    current_prune, peak_prune = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    df.at[i, reward_index] = n
    df.at[i, memory_exact_index_peak] = round((peak_prune / 1024 / 1024), 3)
    df.at[i, memory_train_scratch_index_peak] = round((memory_train / 1024 / 1024), 3)
    df.at[i, memory_tasks_index_peak] = round((memory_task / 1024 / 1024), 3)


    df.to_csv(csv_file_name, index=False, header=header)



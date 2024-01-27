from env.init_query_refine import init_query_refine_2
from train_q_qr import train_q_qr
from inference_q_qr import inference_q_qr
from pruning_synthetic_QR import run_pruning_qr
import pandas as pd
import psutil
import tracemalloc


def memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)  # Convert to megabytes


#input
n_episodes = 4
max_steps_per_episode = 4
learning_rate = 0.1
discount_factor = 0.99
agent_type = "QLearning"
embed_size = 7
num_synthetic_policies = 12

synthetic_reward = [i for i in range(4, num_synthetic_policies+1, 2)]

#output
memories_exact = []
memories_train_scratch = []

csv_file_name = "Memory_" + agent_type + "_QR.csv"


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
    envs[f"grid_world_{i}"] = init_query_refine_2(f"R{i}", embed_size)
    policy_environments[f"policy_R{i}_environments"].append(envs[f"grid_world_{i}"])
env_combined = init_query_refine_2("combined_synthetic", embed_size)
combined_environments.append(env_combined)

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
        time, dag = train_q_qr(env=synthetic_env[f"R{j}_env"][0], n_episodes=n_episodes,
                                      max_steps_per_episode=max_steps_per_episode, agent_type=agent_type,
                                      output_path=q_tables[f"q_table_{j}_output_path"],
                                      learning_rate=learning_rate, discount_factor=discount_factor)
        lstDAG.append(dag)
        current_train, peak_train = tracemalloc.get_traced_memory()
        memory_task += peak_train
    tracemalloc.stop()

    combined_env = combined_environments[0]

    tracemalloc.start()
    time_combined, dag_combined = train_q_qr(env=combined_env[0], n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, agent_type=agent_type, output_path=q_table_combined_output_path, learning_rate=learning_rate, discount_factor=discount_factor)
    current_scratch_train, peak_scratch_train = tracemalloc.get_traced_memory()
    memory_train += peak_scratch_train

    inference_time_combined, reward_ground_truth, _ = inference_q_qr(env=combined_env[0], q_table_path=q_table_combined_output_path, edge_dict=dag_combined)

    current_scratch_infer, peak_scratch_infer = tracemalloc.get_traced_memory()
    memory_train += peak_scratch_infer

    tracemalloc.stop()

    tracemalloc.start()

    best_path, max_reward, total_time, pruning_percentage = run_pruning_qr(env=combined_env[0], dags=lstDAG, discount_factor=discount_factor, learning_rate=learning_rate, number_of_episodes=n_episodes)

    current_prune, peak_prune = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    df.at[i, reward_index] = n
    df.at[i, memory_exact_index_peak] = round((peak_prune / 1024 / 1024), 3)
    df.at[i, memory_train_scratch_index_peak] = round((memory_train / 1024 / 1024), 3)
    df.at[i, memory_tasks_index_peak] = round((memory_task / 1024 / 1024), 3)


    df.to_csv(csv_file_name, index=False, header=header)



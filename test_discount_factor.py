from env.init_gridworld import init_gridworld_1
from train_q_policy import train_q_policy
from DAG import DAG
from utilities import plot_discount_factors

def set_discount_factors(n):
    discount_factors = [0]
    for i in range(1, n):
        discount_factors.append(discount_factors[i - 1] + (1 / experiments_count))
    return discount_factors

# This variable is the input of this file
experiments_count = 10
policy_1 = "path"
policy_2 = "gold"
policy_3 = "combined"
n_episodes = 1000
max_steps_per_episode = 100
learning_rate = 0.1
agent_type = "QLearning"
output_path_1 = "q_table_path.npy"
output_path_2 = "q_table_gold.npy"
output_path_3 = "q_table_combined.npy"

# This variable is the output of this file
pruning_percentages = []
discount_factors = set_discount_factors(experiments_count)

for i in range(experiments_count):
    df = discount_factors[i]
    gridworld_1 = init_gridworld_1(policy_1)
    gridworld_2 = init_gridworld_1(policy_2)
    gridworld_3 = init_gridworld_1(policy_3)
    total_time_1, dag_1 = train_q_policy(gridworld_1, n_episodes, max_steps_per_episode, agent_type, output_path_1, discount_factor=df, learning_rate=learning_rate)
    total_time_2, dag_2 = train_q_policy(gridworld_2, n_episodes, max_steps_per_episode, agent_type, output_path_2, discount_factor=df, learning_rate=learning_rate)
    union_dag = dag_1.union(dag_2)
    min_iters, max_iters = union_dag.min_max_iter() 
    lower_bounds, upper_bounds = union_dag.backtrack(min_iters, max_iters, discount_factor=df, learning_rate = learning_rate)
    pruned_graph, pruning_percentage = union_dag.prune(lower_bounds, upper_bounds)
    pruning_percentages.append(pruning_percentage)
    #pruned_graph.find_paths()

plot_discount_factors(discount_factors, pruning_percentages)
print("Experiment results:\n")
print("Discount factors: " + str(discount_factors))
print("Pruning Percentages: " + str(pruning_percentages))
print("*****************************")


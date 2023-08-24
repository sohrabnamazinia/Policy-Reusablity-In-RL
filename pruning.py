from env.init_gridworld import init_gridworld_1
from train_q_policy import train_q_policy


reward_system_1 = "path"
reward_system_2 = "gold"
grid_world_1 = init_gridworld_1(reward_system_1)
grid_world_2 = init_gridworld_1(reward_system_2)
output_path_1 = "q_table_path.npy"
output_path_2 = "q_table_gold.npy"

n_episodes = 1000
max_steps_per_episode = 100
agent_type = "QLearning"
learning_rate = 0.1
discount_factor = 0.99

# train the agent
total_time_1, dag_1 = train_q_policy(grid_world_1, n_episodes, max_steps_per_episode, agent_type, output_path_1, learning_rate, discount_factor)
total_time_1, dag_2 = train_q_policy(grid_world_2, n_episodes, max_steps_per_episode, agent_type, output_path_2, learning_rate, discount_factor)
union_dag = dag_1.union(dag_2)
max_iterations, min_iterations = union_dag.min_max_iter()
print("Max iteration:\nExample: i: [a, b] means node #i has max_iter = a for action = right, and max_iter = b for action down\n" + str(max_iterations))
print("Min iteration:\nExample: i: [a, b] means node #i has min_iter = a for action = right, and min_iter = b for action down\n" + str(min_iterations))
#union_dag.print(mode=3)
lower_bounds, upper_bounds = union_dag.backtrack(min_iterations, max_iterations, learning_rate, discount_factor)
print("Upper bounds:\nExample: i: [a, b] means node #i has upper_bound = a for action = right, and upper_bound = b for action down\n" + str(upper_bounds))
print("Lower bounds:\nExample: i: [a, b] means node #i has lower_bound = a for action = right, and lower_bound = b for action down\n" + str(lower_bounds))
pruned_graph = union_dag.prune(lower_bounds, upper_bounds)
paths = union_dag.find_paths()
print("Count of resulting paths: " + str(len(paths)))
print("Resulting paths:\n" + str(paths))

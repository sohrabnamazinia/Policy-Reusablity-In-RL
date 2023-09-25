from env.init_gridworld import init_gridworld_1
from train_q_policy import train_q_policy
import time

def compute_pruning(before, after):
    reduced_edge_count = before - after
    return (100 * reduced_edge_count) / before

def get_best_path(gridworld, dag, paths):
    best_path = None
    max_reward = 0
    for path in paths:
        reward = 0
        gridworld.reset()
        for i in range(len(path) - 1):
            state_index_1 = path[i]
            state_index_2 = path[i + 1]
            action = dag.obtain_action(state_index_1, state_index_2)
            grid, r, done, _ = gridworld.step(action)
            reward += r
        if reward >= max_reward:
            max_reward = reward
            best_path = path
    return best_path, max_reward

def run_pruning(gridworld, dag_1, dag_2, learning_rate, discount_factor):
    start_time = time.time()
    union_dag = dag_1.union(dag_2)
    print("Union DAG:")
    union_dag.print()
    max_iterations, min_iterations = union_dag.min_max_iter()
    print("Max iteration:\nExample: i: [a, b] means node #i has max_iter = a for action = right, and max_iter = b for action down\n" + str(max_iterations))
    print("Min iteration:\nExample: i: [a, b] means node #i has min_iter = a for action = right, and min_iter = b for action down\n" + str(min_iterations))
    lower_bounds, upper_bounds = union_dag.backtrack(min_iterations, max_iterations, learning_rate, discount_factor)
    print("Upper bounds:\nExample: i: [a, b] means node #i has upper_bound = a for action = right, and upper_bound = b for action down\n" + str(upper_bounds))
    print("Lower bounds:\nExample: i: [a, b] means node #i has lower_bound = a for action = right, and lower_bound = b for action down\n" + str(lower_bounds))
    edge_count_before_prune = union_dag.graph.number_of_edges()
    pruned_graph, pruning_percentage = union_dag.prune(lower_bounds, upper_bounds)
    print("Pruned Graph:")
    print(pruned_graph)
    paths = union_dag.find_paths()
    print("Count of resulting paths: " + str(len(paths)))
    print("Resulting paths:\n" + str(paths))
    print("Pruning Percentage: %" + str(pruning_percentage))
    total_time = time.time() - start_time
    best_path, max_reward = get_best_path(gridworld=gridworld, dag=union_dag, paths=paths)
    return best_path, max_reward, total_time, pruning_percentage



reward_system_1 = "path"
reward_system_2 = "gold"
grid_world_1 = init_gridworld_1(reward_system_1)
grid_world_2 = init_gridworld_1(reward_system_2)
output_path_1 = "q_table_path.npy"
output_path_2 = "q_table_gold.npy"
n_episodes = 1000
max_steps_per_episode = 100
agent_type = "Sarsa"
learning_rate = 0.1
discount_factor = 0.99

# train the agent and run the algorithm
total_time_1, dag_1, _ = train_q_policy(grid_world_1, n_episodes, max_steps_per_episode, agent_type, output_path_1, learning_rate, discount_factor)
total_time_2, dag_2, _ = train_q_policy(grid_world_2, n_episodes, max_steps_per_episode, agent_type, output_path_2, learning_rate, discount_factor)
print("Total training time - Path: " + str(total_time_1))
print("Total training time - Gold: " + str(total_time_2))
print("Dag of Training - Path:")
dag_1.print(mode=1)
print("Dag of Training - Gold:")
dag_2.print(mode=1)
best_path, max_reward, total_time, pruning_percentage = run_pruning(grid_world_1, dag_1, dag_2, learning_rate, discount_factor)
print("Total time of the pruning algorithm: " + str(total_time))
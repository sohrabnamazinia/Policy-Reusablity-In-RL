from env.init_query_refine import init_query_refine_1
from train_q_qr import train_q_qr
import time
from DAG_qr import DAG

def compute_pruning(before, after):
    reduced_edge_count = before - after
    return (100 * reduced_edge_count) / before

def get_best_path(env, dag, paths):
    best_path = None
    max_reward = 0
    for path in paths:
        reward = 0
        env.reset()
        for i in range(len(path) - 1):
            state_index_1 = path[i]
            state_index_2 = path[i + 1]
            action, r = dag.obtain_action(state_index_1, state_index_2)
            #grid, r, done, _ = env.step(action)
            reward += r
        if reward >= max_reward:
            max_reward = reward
            best_path = path
    return best_path, max_reward

def run_pruning_qr(env, dags, learning_rate, discount_factor, number_of_episodes):
    start_time = time.time()
    union_dag = DAG.union_of_graphs(env, dags, number_of_episodes)
    print("Union DAG:")
    union_dag.print()
    max_iterations, min_iterations = union_dag.min_max_iter()
    #print("Max iteration:\nExample: i: [a, b] means node #i has max_iter = a for action = right, and max_iter = b for action down\n" + str(max_iterations))
    #print("Min iteration:\nExample: i: [a, b] means node #i has min_iter = a for action = right, and min_iter = b for action down\n" + str(min_iterations))
    lower_bounds, upper_bounds = union_dag.backtrack(min_iterations, max_iterations, learning_rate, discount_factor)
    #print("Upper bounds:\nExample: i: [a, b] means node #i has upper_bound = a for action = right, and upper_bound = b for action down\n" + str(upper_bounds))
    #print("Lower bounds:\nExample: i: [a, b] means node #i has lower_bound = a for action = right, and lower_bound = b for action down\n" + str(lower_bounds))
    edge_count_before_prune = union_dag.graph.number_of_edges()
    pruned_graph, pruning_percentage = union_dag.prune(lower_bounds, upper_bounds)
    #print("Pruned Graph:")
    #print(pruned_graph)
    paths = union_dag.find_paths()
    print("Count of resulting paths: " + str(len(paths)))
    print("Resulting paths:\n" + str(paths))
    print("Pruning Percentage: %" + str(pruning_percentage))
    total_time = time.time() - start_time
    best_path, max_reward = get_best_path(env=env, dag=union_dag, paths=paths)
    # if we are already in the goal state
    if best_path == None:
        return None, env.goal_reward, total_time, pruning_percentage
    return best_path, max_reward, total_time, pruning_percentage


# # set inputs
# reward_system_1 = "closeness"
# reward_system_2 = "feature"
# env_1 = init_query_refine_1(reward_system_1)
# env_2 = init_query_refine_1(reward_system_2)
# n_episodes = 3
# max_steps_per_episode = 10
# agent_type = "QLearning"
# learning_rate = 0.1
# discount_factor = 0.99
# output_path_1 = f"q_table_QR_{reward_system_1}_{agent_type}.npy"
# output_path_2 = f"q_table_QR_{reward_system_2}_{agent_type}.npy"

# # train the agent and run the algorithm
# total_time_1, dag_1 = train_q_qr(env_1, n_episodes, max_steps_per_episode, agent_type, output_path_1, learning_rate, discount_factor)
# total_time_2, dag_2 = train_q_qr(env_2, n_episodes, max_steps_per_episode, agent_type, output_path_2, learning_rate, discount_factor)
# print("Total training time - " + reward_system_1 + ": " + str(total_time_1))
# print("Total training time - " + reward_system_2 + ": " + str(total_time_2))
# print("Dag of Training " + reward_system_1 + ": ")
# dag_1.print()
# print("Dag of Training " + reward_system_2 + ": ")
# dag_2.print()
# best_path, max_reward, total_time, pruning_percentage = run_pruning(env_1, dag_1, dag_2, learning_rate, discount_factor)
# print("Total time of the pruning algorithm: " + str(total_time))
# print("Best path: " + str(best_path))
# print("Max reward: " + str(max_reward))
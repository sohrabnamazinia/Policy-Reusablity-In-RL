import numpy as np
from env.init_query_refine import init_query_refine_2
import time

def step_heuristic(current_state, action, dag):
    if not (current_state, action) in dag.edge_dict:
        return None
    next_state, _ = dag.edge_dict[(current_state, action)]
    return next_state

def compute_reward_path(env, path, dag_closeness):
    reward = 0
    state_index = env.get_state_index()
    for i in range(len(path)):
        action = path[i]
        if not (state_index, action) in dag_closeness.edge_dict.keys():
            return 0
        r, next_state_index = dag_closeness.edge_dict[(state_index, action)]  
        reward += r
        state_index = next_state_index
    return reward

def compute_best_path(env, paths, dag_closeness):
    best_path = None
    best_path_reward = 0
    for path in paths:
        reward = compute_reward_path(env, path, dag_closeness)
        if reward > best_path_reward:
            best_path_reward = reward
            best_path = path
    return best_path, best_path_reward

def heuristic(dags, env, k, n, max_allowed_path_size):
    start_time = time.time()
    paths = []
    shortest_paths = []
    stack = []
    shortest_paths_length = float('inf')
    env.reset()
    start_state = env.get_state_index()
    for i in range(n):
        stack.append((start_state, [], f"R{i}"))

    i = 0
    while (len(stack) > 0):
        current_state, current_path, use_policy = stack.pop()
        if (current_state == env.final_state_index) and (current_path not in paths):
            paths.append(current_path)
            if (len(current_path) <= shortest_paths_length):
                shortest_paths.append(current_path)
                shortest_paths_length = len(current_path)
        else:
            state_index = env.get_state_index()
            try:
                q_values = dags[int(use_policy[1:])].edge_dict[state_index]  # Get Q-values for current state from Q-table 1
            except Exception as e:
                continue
            
            # Select the top k actions based on Q-values
            top_k_actions = np.argsort(q_values)[::-1][:k]
            
            for action in top_k_actions:
                # NOTE: escaping actions that cause cycle
                next_state = step_heuristic(env, current_state, action)
                if (next_state == current_state):
                     continue
                next_path = current_path + [action]
                if len(next_path) <= max_allowed_path_size:
                    for i in range(n):
                        stack.append((next_state, next_path, f"R{i}"))

        i += 1
        

    total_time = time.time() - start_time
    best_path, max_cumulative_reward = compute_best_path(env, paths, dag_closeness=dags[0])
    return max_cumulative_reward, best_path, paths, shortest_paths, total_time


def run_heuristic(dags, k, n, max_allowed_path_size, reward_system=None, env=None):

    # q_tables = {}

    # for i in range(n):
    #     path = q_tables_path[f"q_table_{i}_output_path"]
    #     q_tables[f"R{i}"] = np.load(path)

    if reward_system != None:
        env = init_query_refine_2(reward_system)

    return heuristic(dags, env, k, n, max_allowed_path_size)

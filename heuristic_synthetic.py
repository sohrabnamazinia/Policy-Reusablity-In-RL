import numpy as np
from env.init_gridworld import init_gridworld_1
import time

def step_heuristic(grid_world, current_state, action):
    next_state = [current_state[0], current_state[1]]

    if action == 0:   # right
            next_state[1] += 1
    elif action == 1: # down
            next_state[0] += 1

    next_state = np.clip(next_state, (0, 0), (grid_world.grid_width - 1, grid_world.grid_length - 1))
    return next_state.tolist()

def compute_reward_path(env, path):
    env.reset()
    reward = 0
    for i in range(len(path)):
        action = path[i]
        grid, r, done, _ = env.step(action)
        reward += r
    return reward

def compute_best_path(env, paths):
    best_path = None
    best_path_reward = 0
    for path in paths:
        reward = compute_reward_path(env, path)
        if reward > best_path_reward:
            best_path_reward = reward
            best_path = path
    return best_path, best_path_reward

def heuristic(q_tables, env, k, n, max_allowed_path_size):
    start_time = time.time()
    paths = []
    shortest_paths = []
    stack = []
    shortest_paths_length = float('inf')
    env.reset()
    start_state = env.start_position
    for i in range(n):
        stack.append((start_state, [], f"R{i}"))

    i = 0
    while (len(stack) > 0):
        current_state, current_path, use_policy = stack.pop()
        if (current_state == env.target_position) and (current_path not in paths):
            paths.append(current_path)
            if (len(current_path) <= shortest_paths_length):
                shortest_paths.append(current_path)
                shortest_paths_length = len(current_path)
        else:
            state_index = env.state_to_index(env.agent_position)
            q_values = q_tables[use_policy][state_index]  # Get Q-values for current state from Q-table 1

            
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
    best_path, max_cumulative_reward = compute_best_path(env, paths)
    return max_cumulative_reward, best_path, paths, shortest_paths, total_time


def run_heuristic(q_tables_path, k, n, max_allowed_path_size, reward_system=None, gridworld=None):

    q_tables = {}

    for i in range(n):
        path = q_tables_path[f"q_table_{i}_output_path"]
        q_tables[f"R{i}"] = np.load(path)

    if reward_system != None:
        gridworld = init_gridworld_1(reward_system)

    return heuristic(q_tables, gridworld, k, n, max_allowed_path_size)

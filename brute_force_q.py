import numpy as np
from env.init_gridworld import init_gridworld_1

def step_brute_force(grid_world, current_state, action):
    next_state = [current_state[0], current_state[1]]
    if action == 0:   # up
            next_state[0] -= 1
    elif action == 1: # right
            next_state[1] += 1
    elif action == 2: # down
            next_state[0] += 1
    elif action == 3: # left
            next_state[1] -= 1

    next_state = np.clip(next_state, (0, 0), (grid_world.grid_width - 1, grid_world.grid_length - 1))
    return next_state.tolist()

def brute_force_q(q_table_1, q_table_2, env, k, max_allowed_path_size):
    paths = []
    shortest_paths = []
    stack = []
    shortest_paths_length = float('inf')
    start_state = env.start_position
    stack.append((start_state, [], True))
    stack.append((start_state, [], False))

    while (len(stack) > 0):
        current_state, current_path, use_policy1 = stack.pop()
        if (current_state == env.target_position) and (current_path not in paths):
            paths.append(current_path)
            if (len(current_path) <= shortest_paths_length):
                shortest_paths.append(current_path)
                shortest_paths_length = len(current_path)
        else:
            state_index = np.ravel_multi_index(tuple(env.agent_position), dims=env.grid.shape)
            if use_policy1:
                q_values = q_table_1[state_index]  # Get Q-values for current state from Q-table 1
            else:
                q_values = q_table_2[state_index]  # Get Q-values for current state from Q-table 2
            
            # Select the top k actions based on Q-values
            top_k_actions = np.argsort(q_values)[::-1][:k]
            
            for action in top_k_actions:
                # NOTE: escaping actions that cause cycle
                if action in [0, 3]:
                     continue
                next_state = step_brute_force(env, current_state, action)
                if (next_state == current_state):
                     continue
                next_path = current_path + [action]
                if len(next_path) <= max_allowed_path_size:
                    stack.append((next_state, next_path, use_policy1))
                    stack.append((next_state, next_path, not use_policy1))

    return paths, shortest_paths


def run_brute_force_q(q_table_1_path, q_table_2_path, reward_system, k, max_allowed_path_size):
    q_table_1 = np.load(q_table_1_path)
    q_table_2 = np.load(q_table_2_path)
    grid_world = init_gridworld_1(reward_system)
    return brute_force_q(q_table_1, q_table_2, grid_world, k, max_allowed_path_size)


# set up inputs
reward_system = "combined"
q_table_1_path = "q_table_path.npy"
q_table_2_path = "q_table_gold.npy"
k = 1
max_allowed_path_size = 8

paths, shortest_paths = run_brute_force_q(q_table_1_path, q_table_2_path, reward_system, k, max_allowed_path_size)
print("All paths count:\n" + str(len(paths)))
print(paths)
print("Shortest paths count:\n" + str(len(shortest_paths)))
print(shortest_paths)
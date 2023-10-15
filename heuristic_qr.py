import numpy as np
from env.init_query_refine import init_query_refine_1
import time
from train_q_qr import train_q_qr

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

def heuristic(q_table_1, q_table_2, env, k, max_allowed_path_size, dag_closeness, dag_feature):
    start_time = time.time()
    paths = []
    shortest_paths = []
    stack = []
    shortest_paths_length = float('inf')
    env.reset()
    start_state = env.get_state_index()
    stack.append((start_state, [], True))
    stack.append((start_state, [], False))
    i = 0
    while (len(stack) > 0):
        current_state, current_path, use_policy1 = stack.pop()
        if (current_state == env.final_state_index) and (current_path not in paths):
            paths.append(current_path)
            if (len(current_path) <= shortest_paths_length):
                shortest_paths.append(current_path)
                shortest_paths_length = len(current_path)
        else:
            state_index = env.get_state_index()
            if use_policy1:
                q_values = q_table_1[state_index]  # Get Q-values for current state from Q-table 1
            else:
                q_values = q_table_2[state_index]  # Get Q-values for current state from Q-table 2
            
            # Select the top k actions based on Q-values
            top_k_actions = np.argsort(q_values)[::-1][:k]
            
            for action in top_k_actions:
                # NOTE: escaping actions that cause cycle
                # if action in [0, 3]:
                #      continue
                next_state = None
                if (use_policy1):
                    next_state = step_heuristic(current_state, action, dag_closeness)
                elif (not use_policy1):
                    next_state = step_heuristic(current_state, action, dag_feature)
                if next_state == None:
                    continue
                if (next_state == current_state):
                     continue
                next_path = current_path + [action]
                if len(next_path) <= max_allowed_path_size:
                    stack.append((next_state, next_path, use_policy1))
                    stack.append((next_state, next_path, not use_policy1))
        i += 1
        

    total_time = time.time() - start_time
    best_path, max_cumulative_reward = compute_best_path(env, paths, dag_closeness)
    return max_cumulative_reward, best_path, paths, shortest_paths, total_time


def run_heuristic(q_table_1_path, q_table_2_path, k, max_allowed_path_size, dag_closeness, dag_feature, reward_system=None, env=None):
    q_table_1 = np.load(q_table_1_path)
    q_table_2 = np.load(q_table_2_path)
    if reward_system != None:
        env = init_query_refine_1(reward_system)
    return heuristic(q_table_1, q_table_2, env, k, max_allowed_path_size, dag_closeness, dag_feature)


# set up inputs
reward_system = "combined"
output_path_closeness = "q_table_qr_closeness.npy"
output_path_feature = "q_table_qr_feature.npy"
agent_type = "QLearning"
k = 2
n_episodes = 1
env_closeness = init_query_refine_1("closeness")
env_feature = init_query_refine_1("feature")
max_steps_per_episode = 8
_, dag_closeness = train_q_qr(env_closeness, n_episodes, max_steps_per_episode, agent_type, output_path_closeness)
_, dag_feature = train_q_qr(env_feature, n_episodes, max_steps_per_episode, agent_type, output_path_feature)

max_cumulative_reward, best_path, paths, shortest_paths, total_time = run_heuristic(output_path_closeness, output_path_feature, k, max_steps_per_episode, reward_system=reward_system, dag_closeness=dag_closeness, dag_feature=dag_feature)
print("Total_time: " + str(total_time))
print("All paths count:\n" + str(len(paths)))
#print(paths)
print("Shortest paths count:\n" + str(len(shortest_paths)))
print("Max cumulative reward: " + str(max_cumulative_reward))
#print(shortest_paths)
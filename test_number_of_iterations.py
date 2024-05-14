from train_q_policy import train_q_policy
from env.init_gridworld import init_gridworld_1
import numpy as np

reward1 = "path"
reward2 = "gold"
reward3 = "combined"
grid_world_1 = init_gridworld_1(reward1)
grid_world_2 = init_gridworld_1(reward2)
grid_world_3 = init_gridworld_1(reward3)

n_episodes = 1000
max_steps_per_episode = 100
agent_type = "QLearning"
output_path = "test_number_of_iterations.npy"

_, _, _, visited_count_transitions_1 = train_q_policy(grid_world_1, n_episodes, max_steps_per_episode, agent_type, output_path)
print(visited_count_transitions_1)
_, _, _, visited_count_transitions_2 = train_q_policy(grid_world_2, n_episodes, max_steps_per_episode, agent_type, output_path)
print(visited_count_transitions_2)
_, _, _, visited_count_transitions_3 = train_q_policy(grid_world_3, n_episodes, max_steps_per_episode, agent_type, output_path)
print(visited_count_transitions_3)

shape = visited_count_transitions_1.shape
comparison_results = np.empty(shape, dtype=object)

for i in range(shape[0]):
    for j in range(shape[1]):
        for k in range(shape[2]):
            val1 = visited_count_transitions_1[i, j, k]
            val2 = visited_count_transitions_2[i, j, k]
            val3 = visited_count_transitions_3[i, j, k]
            summation = val1 + val2
            
            if val3 < summation:
                comparison_results[i, j, k] = 'smaller'
            elif val3 > summation:
                comparison_results[i, j, k] = 'bigger'
            else:
                comparison_results[i, j, k] = 'equal'

print(comparison_results)
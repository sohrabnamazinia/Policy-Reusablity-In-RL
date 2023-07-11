import numpy as np
from gridworld import GridWorld

# Define gold and block positions
gold_positions = [[1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [5, 4], [5, 5], [5, 6]]
block_positions = [[2, 2], [3, 4], [5, 5], [6, 6], [7, 7], [8, 8]]

# Instantiate GridWorld
grid_world = GridWorld(grid_size=10, gold_positions=gold_positions, block_positions=block_positions)

# Flatten the grid to get the total number of states
n_states = np.product(grid_world.grid.shape)

# Get the total number of actions
n_actions = len(grid_world.action_space)

# Initialize the Q-Learning agent
q_agent = QLearningAgent(n_states=n_states, n_actions=n_actions)

# Train the Q-Learning agent
n_episodes = 10000
max_steps_per_episode = 100

for episode in range(n_episodes):
    state = grid_world.reset().flatten()
    state_index = np.ravel_multi_index(tuple(state), dims=grid_world.grid.shape)

    for step in range(max_steps_per_episode):
        action = q_agent.get_action(state_index)

        next_state, reward, done, _ = grid_world.step(action)
        next_state_index = np.ravel_multi_index(tuple(next_state.flatten()), dims=grid_world.grid.shape)

        q_agent.update_q_table(state_index, action, reward, next_state_index)

        state_index = next_state_index

        if done:
            break

# Save the q_table for future use
np.save('q_table.npy', q_agent.q_table)

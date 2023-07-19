import numpy as np
from agents.q_agent import QLearningAgent
from env.gridworld import GridWorld

# Define gold and block positions
gold_positions = [[0, 2], [2, 2], [2, 5], [4, 1]]
block_positions = []
reward_system = "path"
agent_initial_position = [0, 0]
target_position = [4, 4]
cell_low_value = -1
cell_high_value = 10
start_position_value = 5
target_position_value = 10

# Instantiate GridWorld
grid_world = GridWorld(grid_width=5, grid_length=6, gold_positions=gold_positions, block_positions=block_positions
                       , reward_system="path", agent_position=agent_initial_position, target_position=target_position
                       , cell_high_value=cell_high_value, cell_low_value=cell_low_value,
                       start_position_value=start_position_value, target_position_value=target_position_value)

# Flatten the grid to get the total number of states
n_states = np.product(grid_world.grid.shape)

# Get the total number of actions
n_actions = grid_world.action_space.n

# Initialize the Q-Learning agent
q_agent = QLearningAgent(n_states=n_states, n_actions=n_actions)

# Train the Q-Learning agent
n_episodes = 10000
max_steps_per_episode = 100

for episode in range(n_episodes):
    grid_world.reset().flatten()
    state_index = np.ravel_multi_index(tuple(grid_world.agent_position), dims=grid_world.grid.shape)

    for step in range(max_steps_per_episode):
        action = q_agent.get_action(state_index)

        grid, reward, done, _ = grid_world.step(action)
        next_state_index = np.ravel_multi_index(tuple(grid_world.agent_position.flatten()), dims=grid_world.grid.shape)

        q_agent.update_q_table(state_index, action, reward, next_state_index)

        state_index = next_state_index

        if done:
            break

# Save the q_table for future use
np.save('q_table.npy', q_agent.q_table)

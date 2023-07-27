import numpy as np
from agents.q_agent import QLearningAgent
from agents.q_agent import SarsaAgent
from env.init_gridworld import init_gridworld_1

def train_q_policy(grid_world, n_episodes, max_steps_per_episode, agent_type):

    # Flatten the grid to get the total number of states
    n_states = np.product(grid_world.grid.shape)

    # Get the total number of actions
    n_actions = grid_world.action_space.n

    # Initialize the Q-Learning agent
    q_agent = None
    if agent_type == "QLearning":
        q_agent = QLearningAgent(n_states=n_states, n_actions=n_actions)
    elif agent_type == "Sarsa":
        q_agent = SarsaAgent(n_states=n_states, n_actions=n_actions)
    

    for episode in range(n_episodes):
        grid_world.reset().flatten()
        state_index = np.ravel_multi_index(tuple(grid_world.agent_position), dims=grid_world.grid.shape)

        for step in range(max_steps_per_episode):
            action = q_agent.get_action(state_index)

            grid, reward, done, _ = grid_world.step(action)
            next_state_index = np.ravel_multi_index(tuple(grid_world.agent_position.flatten()), dims=grid_world.grid.shape)

            if agent_type == "Sarsa":
                next_action = q_agent.get_action(next_state_index)
                q_agent.update_q_table(state_index, action, reward, next_state_index, next_action)
            elif agent_type == "QLearning":
                q_agent.update_q_table(state_index, action, reward, next_state_index)

            state_index = next_state_index

            if done:
                break

        # update lerning rate and explortion rate
        q_agent.learning_rate = max(q_agent.learning_rate * q_agent.learning_rate_decay, q_agent.min_learning_rate)
        q_agent.exploration_rate = max(q_agent.exploration_rate * q_agent.exploration_rate_decay, q_agent.min_exploration_rate)

    # Save the q_table for future use
    np.save('q_table.npy', q_agent.q_table)



# Define env and train parameters
reward_system = "path"
grid_world = init_gridworld_1(reward_system)
n_episodes = 10000
max_steps_per_episode = 100
agent_type = "QLearning"

# train the agent
train_q_policy(grid_world, n_episodes, max_steps_per_episode, agent_type)

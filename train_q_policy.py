import numpy as np
from agents.q_agent import QLearningAgent
from agents.q_agent import SarsaAgent
from env.init_gridworld import init_gridworld_1
from DAG import DAG
import wandb
import time
import pandas as pd
from utilities import plot_cummulative_reward



def train_q_policy(grid_world, n_episodes, max_steps_per_episode, agent_type, output_path, learning_rate=None, discount_factor=None, result_step_size=1, plot_cumulative_reward=False):

    # Flatten the grid to get the total number of states
    n_states = np.prod(grid_world.grid.shape)

    # Get the total number of actions
    n_actions = grid_world.action_space.n

    dag = DAG(gridworld=grid_world, N=n_episodes)

    # Initialize the Q-Learning agent
    q_agent = None
    if agent_type == "QLearning":
        q_agent = QLearningAgent(n_states=n_states, n_actions=n_actions)
    elif agent_type == "Sarsa":
        q_agent = SarsaAgent(n_states=n_states, n_actions=n_actions)

    # check if we want to hardcode lr and df by using input parameters
    if learning_rate != None:
        q_agent.learning_rate = learning_rate
    if discount_factor != None:
        q_agent.discount_factor = discount_factor

    df = pd.DataFrame()
    csv_index_episode = 0
    csv_index_cummulative_reward = 1

    header = ["Episode", "Cumulative Reward"]
    cumulative_reward = 0
    total_time = 0
    

    for episode in range(n_episodes):
        # turn on stopwatch
        start_time = time.time()

        grid_world.reset().flatten()
        state_index = grid_world.state_to_index(grid_world.agent_position)

        for step in range(max_steps_per_episode):
            grid_world.visited[grid_world.agent_position[0]][grid_world.agent_position[1]] += 1
            action = q_agent.get_action(state_index)

            grid, reward, done, info = grid_world.step(action)

            cumulative_reward += reward
            next_state_index = grid_world.state_to_index(grid_world.agent_position)
            
            if agent_type == "Sarsa":
                next_action = q_agent.get_action(next_state_index)
                q_agent.update_q_table(state_index, action, reward, next_state_index, next_action)
            elif agent_type == "QLearning":
                q_agent.update_q_table(state_index, action, reward, next_state_index)

            if (state_index != next_state_index):
                dag.add_edge(state_index, next_state_index)
            state_index = next_state_index

            if done:
                break

        # update lerning rate and explortion rate
        q_agent.exploration_rate = max(q_agent.exploration_rate * q_agent.exploration_rate_decay, q_agent.min_exploration_rate)

        # turn of stopwatch
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        # log cumulative reward

        if episode % result_step_size == 0:
            df.at[(episode / result_step_size) + 1, csv_index_episode] = episode
            df.at[(episode / result_step_size) + 1, csv_index_cummulative_reward] = cumulative_reward

    
    # Save the q_table for future use
    csv_file_name = "Train_" + grid_world.reward_system + "_" + agent_type + "_" + str(n_episodes) + ".csv"
    df.to_csv(csv_file_name, index=False, header=header)
    if plot_cumulative_reward:
        plot_cummulative_reward(csv_file_name, header[0], header[1])
    np.save(output_path, q_agent.q_table)
    #run.finish()

    return total_time, dag, cumulative_reward

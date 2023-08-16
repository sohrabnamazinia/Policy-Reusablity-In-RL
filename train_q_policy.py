import numpy as np
from agents.q_agent import QLearningAgent
from agents.q_agent import SarsaAgent
from env.init_gridworld import init_gridworld_1
from DAG import DAG
import wandb
import time

def train_q_policy(grid_world, n_episodes, max_steps_per_episode, agent_type, output_path):

    # Flatten the grid to get the total number of states
    n_states = np.product(grid_world.grid.shape)

    # Get the total number of actions
    n_actions = grid_world.action_space.n

    dag = DAG(n_states, n_actions, n_episodes, grid_world.state_to_index(grid_world.target_position), grid_world.grid_length)

    # Initialize the Q-Learning agent
    q_agent = None
    if agent_type == "QLearning":
        q_agent = QLearningAgent(n_states=n_states, n_actions=n_actions)
    elif agent_type == "Sarsa":
        q_agent = SarsaAgent(n_states=n_states, n_actions=n_actions)

    run = wandb.init(project="Train_Q")

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
            # if (False in info):
            #     continue
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
        q_agent.learning_rate = max(q_agent.learning_rate * q_agent.learning_rate_decay, q_agent.min_learning_rate)
        q_agent.exploration_rate = max(q_agent.exploration_rate * q_agent.exploration_rate_decay, q_agent.min_exploration_rate)

        # turn of stopwatch
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        # log cumulative reward
        wandb.log({"Cumulative Reward": cumulative_reward}, step=episode)
        wandb.log({"Total Training Time": total_time}, step=episode)


    # Save the q_table for future use
    np.save(output_path, q_agent.q_table)
    run.finish()

    return total_time, dag

# Define env and train parameters
# reward_system = "combined"
# grid_world = init_gridworld_1(reward_system)
# n_episodes = 1000
# max_steps_per_episode = 100
# agent_type = "QLearning"
# output_path = "q_table_combined.npy"

# # train the agent
# total_time, dag = train_q_policy(grid_world, n_episodes, max_steps_per_episode, agent_type, output_path)
# dag.print(mode=3, env_length=grid_world.grid_length)
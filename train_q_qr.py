from env.init_query_refine import init_query_refine_1
import numpy as np
from agents.q_agent import QLearningAgent, SarsaAgent
import math

def train_q_qr(env, n_episodes, max_steps_per_episode, agent_type, output_path, learning_rate=None, discount_factor=None):

    n_states = int(math.pow(2, env.query_vector.shape[0]))
    n_actions = env.action_space.n

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
    
    cumulative_reward = 0
    for episode in range(n_episodes):
        env.reset().flatten()
        state_index = env.get_state_index()

        for step in range(max_steps_per_episode):
            action = q_agent.get_action(state_index)

            grid, reward, done, info = env.step(action)
            cumulative_reward += reward
            next_state_index = env.get_state_index()
            
            if agent_type == "Sarsa":
                next_action = q_agent.get_action(next_state_index)
                q_agent.update_q_table(state_index, action, reward, next_state_index, next_action)
            elif agent_type == "QLearning":
                q_agent.update_q_table(state_index, action, reward, next_state_index)

            state_index = next_state_index

            if done:
                break

        # update lerning rate and explortion rate
        q_agent.exploration_rate = max(q_agent.exploration_rate * q_agent.exploration_rate_decay, q_agent.min_exploration_rate)

    np.save(output_path, q_agent.q_table)

    return
    

# test a sample
reward_system = "closeness"
env = init_query_refine_1(reward_system)
n_episodes = 1000
max_steps_per_episode = 100
agent_type = "QLearning"
output_path = "Train_QR_" + agent_type + "_" + str(n_episodes) + ".npy"
train_q_qr(env, n_episodes=n_episodes, max_steps_per_episode=100, output_path=output_path, agent_type=agent_type)
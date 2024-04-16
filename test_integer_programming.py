from env.init_gridworld import init_gridworld_1
from train_q_policy import train_q_policy
from pruning import run_pruning_IP

# set inputs
reward_system_1 = "path"
reward_system_2 = "gold"
env_1 = init_gridworld_1(reward_system_1)
env_2 = init_gridworld_1(reward_system_2)
n_episodes = 1000
max_steps_per_episode = 10
agent_type = "QLearning"
learning_rate = 0.1
discount_factor = 0.99
output_path_1 = f"q_table_QR_{reward_system_1}_{agent_type}.npy"
output_path_2 = f"q_table_QR_{reward_system_2}_{agent_type}.npy"

# train the agent and run the algorithm
total_time_1, dag_1, c_reward_1 = train_q_policy(env_1, n_episodes, max_steps_per_episode, agent_type, output_path_1, learning_rate, discount_factor)
total_time_2, dag_2, c_reward_2 = train_q_policy(env_2, n_episodes, max_steps_per_episode, agent_type, output_path_2, learning_rate, discount_factor)
print("Total training time - " + reward_system_1 + ": " + str(total_time_1))
print("Total training time - " + reward_system_2 + ": " + str(total_time_2))
print("Dag of Training " + reward_system_1 + ": ")
dag_1.print()
print("Dag of Training " + reward_system_2 + ": ")
dag_2.print()
best_path, max_reward, total_time, pruning_percentage = run_pruning_IP(env_1, dag_1, dag_2, learning_rate, discount_factor, n_episodes)
print("Total time of the pruning algorithm: " + str(total_time))
print("Best path: " + str(best_path))
print("Max reward: " + str(max_reward))
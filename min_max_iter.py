from env.init_gridworld import init_gridworld_1
from train_q_policy import train_q_policy


reward_system_1 = "path"
reward_system_2 = "gold"
grid_world_1 = init_gridworld_1(reward_system_1)
grid_world_2 = init_gridworld_1(reward_system_2)
output_path_1 = "q_table_path.npy"
output_path_2 = "q_table_gold.npy"

n_episodes = 1000
max_steps_per_episode = 100
agent_type = "QLearning"

# train the agent
total_time_1, dag_1 = train_q_policy(grid_world_1, n_episodes, max_steps_per_episode, agent_type, output_path_1)
total_time_1, dag_2 = train_q_policy(grid_world_2, n_episodes, max_steps_per_episode, agent_type, output_path_2)

union_dag = dag_1.union(dag_2)
min_iterations, max_iterations = union_dag.min_max_iter()

print(min_iterations)
union_dag.print(mode=3)

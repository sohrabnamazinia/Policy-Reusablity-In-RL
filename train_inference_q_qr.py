from train_q_qr import train_q_qr
from inference_q_qr import inference_q_qr
from env.init_query_refine import init_query_refine_1

# set inputs
reward_system = "combined"
env = init_query_refine_1(reward_system)
n_episodes = 3
max_steps_per_episode = 10
agent_type = "QLearning"
output_path = f"QTable_QR_{agent_type}_{reward_system}.npy"

# train
total_train_time, dag = train_q_qr(env, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode, output_path=output_path, agent_type=agent_type)
print("total train time: " + str(total_train_time))
dag.print()
print(f"Edge dict of the DAG:\n: {dag.edge_dict}")

# inference
total_inference_time, total_reward, path = inference_q_qr(env, q_table_path=output_path, edge_dict=dag.edge_dict)
print("Total inference time: " + str(total_inference_time))
print("Total reward: " + str(total_reward))
print("Path: " + str(path))
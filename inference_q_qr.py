import numpy as np
from env.query_refine import Query_Refine
from env.init_query_refine import init_query_refine_1
import wandb
import time

# inference method
# the q_table.npy file must be in the same directory as this file
def inference_q_qr(env, q_table_path, edge_dict):

    # Load the Q-table
    q_table = np.load(q_table_path)

    #run = wandb.init(project="Inference_Q")
    total_time = 0
    total_reward = 0

    # Reset the environment to its initial state
    env.reset().flatten()
    state_index = env.get_state_index()

    if state_index == env.final_state_index:
        return 0, env.goal_reward, []

    # Maximum number of steps for inference
    max_steps_inference = 10
    path = []

    for step in range(max_steps_inference):
        # turn on stopwatch
        start_time = time.time()

        # print the current state index of the agent
        print("Agent State: " + str(env.get_state_index()))

        # greedy action selection (inference)
        action = np.argmax(q_table[state_index, :])
        path.append(action)

        # print action
        print("Action: ", action)

        # step
        state_index = env.get_state_index()
        next_state_index, reward = edge_dict[(state_index, action)]
        total_reward += reward

        # upadate state index
        state_index = next_state_index
        done = state_index == env.final_state_index

        # print step index
        print("Step: ", step + 1)

        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        # check if the agent reached the target or the maximum number of steps is reached
        if done:
            if reward > 0:
                print("Agent reached the target!")
            else:
                print("Agent failed to reach the target!")
            break

        #wandb.log({"Total Inference Time": total_time}, step=step)
    
    #run.finish()
    return total_time, total_reward, path



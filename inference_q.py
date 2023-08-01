import numpy as np
from env.gridworld import GridWorld
from env.init_gridworld import init_gridworld_1

# inference method
# the q_table.npy file must be in the same directory as this file
def inference_q(grid_world):
    # Load the Q-table
    q_table = np.load('q_table.npy')
    print(q_table)

    # Reset the environment to its initial state
    grid_world.reset().flatten()
    state_index = np.ravel_multi_index(tuple(grid_world.agent_position), dims=grid_world.grid.shape)

    # Maximum number of steps for inference
    max_steps_inference = 100

    for step in range(max_steps_inference):
        # print the current location of the agent
        print("Agent location: " + str(grid_world.agent_position))

        # greedy action selection (inference)
        action = np.argmax(q_table[state_index, :])

        # print action
        print("Action: ", action)

        # step
        grid, reward, done, _ = grid_world.step(action)
        next_state_index = np.ravel_multi_index(tuple(grid_world.agent_position.flatten()), dims=grid_world.grid.shape)

        # upadate state index
        state_index = next_state_index

        # print step index
        print("Step: ", step + 1)
        print(grid)

        # check if the agent reached the target or the maximum number of steps is reached
        if done:
            if reward > 0:
                print("Agent reached the target!")
            else:
                print("Agent failed to reach the target!")
            break

grid_world = init_gridworld_1("combined")
inference_q(grid_world)

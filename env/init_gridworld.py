from env.gridworld import GridWorld
import random

def init_gridworld_1(reward_system):
    # Define the environment details
    gold_positions = [[0, 2], [2, 2], [2, 5], [4, 1]]
    block_positions = []
    reward_system = reward_system
    agent_initial_position = [0, 0]
    target_position = [4, 4]
    cell_low_value = -1
    cell_high_value = 10
    start_position_value = 5
    target_position_value = 10
    block_position_value = -1
    gold_position_value = +1
    agent_position_value = 7
    block_reward = -10
    target_reward = +100

    # Instantiate GridWorld
    grid_world = GridWorld(grid_width=5, grid_length=6, gold_positions=gold_positions, block_positions=block_positions
                        , reward_system=reward_system, agent_position=agent_initial_position, target_position=target_position
                        , cell_high_value=cell_high_value, cell_low_value=cell_low_value,
                        start_position_value=start_position_value, target_position_value=target_position_value, block_position_value=block_position_value, gold_position_value=gold_position_value, agent_position_value=agent_position_value, block_reward=block_reward, target_reward=target_reward)
    
    return grid_world

def init_gridworld_2(reward_system):
    # Define the environment details
    gold_positions = []
    block_positions = []
    reward_system = reward_system
    agent_initial_position = [0, 0]
    target_position = [50, 50]
    cell_low_value = -1
    cell_high_value = 10
    start_position_value = 5
    target_position_value = 10
    block_position_value = -1
    gold_position_value = +1
    agent_position_value = 7
    block_reward = -10
    target_reward = +100

    # Instantiate GridWorld
    grid_world = GridWorld(grid_width=100, grid_length=100, gold_positions=gold_positions, block_positions=block_positions
                        , reward_system=reward_system, agent_position=agent_initial_position, target_position=target_position
                        , cell_high_value=cell_high_value, cell_low_value=cell_low_value,
                        start_position_value=start_position_value, target_position_value=target_position_value, block_position_value=block_position_value, gold_position_value=gold_position_value, agent_position_value=agent_position_value, block_reward=block_reward, target_reward=target_reward)
    
    return grid_world


def init_gridworld_3(reward_system, grid_width, grid_length):
    # Define the environment details
    gold_positions = []
    for i in range(1, min(grid_width, grid_length) - 1):
        gold_positions.append([i, i])

    block_positions = []
    reward_system = reward_system
    agent_initial_position = [0, 0]
    target_position = [grid_width - 1, grid_length - 1]
    cell_low_value = -1
    cell_high_value = 10
    start_position_value = 5
    target_position_value = 10
    block_position_value = -1
    gold_position_value = +1
    agent_position_value = 7
    block_reward = -10
    target_reward = +100

    # Instantiate GridWorld
    grid_world = GridWorld(grid_width=grid_width, grid_length=grid_length, gold_positions=gold_positions, block_positions=block_positions
                        , reward_system=reward_system, agent_position=agent_initial_position, target_position=target_position
                        , cell_high_value=cell_high_value, cell_low_value=cell_low_value,
                        start_position_value=start_position_value, target_position_value=target_position_value, block_position_value=block_position_value, gold_position_value=gold_position_value, agent_position_value=agent_position_value, block_reward=block_reward, target_reward=target_reward)
    
    return grid_world

def init_gridworld_4(reward_system, width_size, length_size):
    # Define the environment details
    gold_positions = []
    for i in range(1, min(width_size, length_size) - 1):
        gold_positions.append([i, i])

    block_positions = []
    reward_system = reward_system
    agent_initial_position = [0, 0]
    target_position = [width_size - 1, length_size - 1]
    cell_low_value = -1
    cell_high_value = 10
    start_position_value = 5
    target_position_value = 10
    block_position_value = -1
    gold_position_value = +1
    agent_position_value = 7
    block_reward = -10
    target_reward = +100

    # Instantiate GridWorld
    grid_world = GridWorld(grid_width=width_size, grid_length=length_size, gold_positions=gold_positions, block_positions=block_positions
                        , reward_system=reward_system, agent_position=agent_initial_position, target_position=target_position
                        , cell_high_value=cell_high_value, cell_low_value=cell_low_value,
                        start_position_value=start_position_value, target_position_value=target_position_value, block_position_value=block_position_value, gold_position_value=gold_position_value, agent_position_value=agent_position_value, block_reward=block_reward, target_reward=target_reward)
    
    return grid_world

def init_gridworld_5(reward_system, width_size, length_size):
    # Define the environment details
    gold_positions = []
    for i in range(1, min(width_size, length_size)):
        if i != width_size or i != int(length_size / 2):
            gold_positions.append([i, i])

    block_positions = []
    reward_system = reward_system
    agent_initial_position = [0, 0]
    target_position = [int(width_size) - 1, int(length_size / 2)]
    cell_low_value = -1
    cell_high_value = 10
    start_position_value = 5
    target_position_value = 10
    block_position_value = -1
    gold_position_value = +1
    agent_position_value = 7
    block_reward = -10
    target_reward = +100

    # Instantiate GridWorld
    grid_world = GridWorld(grid_width=width_size, grid_length=length_size, gold_positions=gold_positions, block_positions=block_positions
                        , reward_system=reward_system, agent_position=agent_initial_position, target_position=target_position
                        , cell_high_value=cell_high_value, cell_low_value=cell_low_value,
                        start_position_value=start_position_value, target_position_value=target_position_value, block_position_value=block_position_value, gold_position_value=gold_position_value, agent_position_value=agent_position_value, block_reward=block_reward, target_reward=target_reward)
    
    return grid_world

def init_gridworld_6(reward_system, action_size, side_length):
    # Define the environment details
    gold_positions = []
    block_positions = []
    reward_system = reward_system
    agent_initial_position = [0, 0]
    target_position = [3, 3]
    cell_low_value = -1
    cell_high_value = 10
    start_position_value = 5
    target_position_value = 10
    block_position_value = -1
    gold_position_value = +1
    agent_position_value = 7
    block_reward = -10
    target_reward = +100
    action_size = action_size

    # Instantiate GridWorld
    grid_world = GridWorld(grid_width=side_length, grid_length=side_length, gold_positions=gold_positions, block_positions=block_positions
                        , reward_system=reward_system, agent_position=agent_initial_position, target_position=target_position
                        , cell_high_value=cell_high_value, cell_low_value=cell_low_value,
                        start_position_value=start_position_value, target_position_value=target_position_value, block_position_value=block_position_value, gold_position_value=gold_position_value, agent_position_value=agent_position_value, block_reward=block_reward, target_reward=target_reward, action_size=action_size)
    
    return grid_world
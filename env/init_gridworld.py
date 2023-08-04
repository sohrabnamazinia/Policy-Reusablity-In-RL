from env.gridworld import GridWorld

def init_gridworld_1(reward_system):
    # Define the environment details
    gold_positions = [[0, 2], [2, 2], [2, 5], [4, 1]]
    #gold_positions = [[4, 2], [4, 1], [4, 0], [4, 3]]
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
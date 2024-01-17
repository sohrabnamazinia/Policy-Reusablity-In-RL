
import random

def generate_random_policies(state_count, n, l, u):
    
    '''
    
    Input:

    n: Number of synthetic policies.
    l: Lower bound for the reward range.
    u: Upper bound for the reward range.

    Output:

    A dictionary containing n individual dictionaries, each representing a policy. 
    Each policy dictionary contains a list of two random variables, each falling within the range of l to u, 
    for every possible state.
    
    '''
    
    reward_dict = {}

    for outer_key in range(n):
        policy_dict = {}
        for i in range(state_count):
            for j in range(state_count):
                policy_dict[(i, j)] = [round(random.uniform(l, u), 2), round(random.uniform(l, u), 2)]
        reward_dict[outer_key] = policy_dict

    return reward_dict



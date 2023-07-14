import numpy as np

agent_position = np.array([3, 4])
target_position = [4, 4]
done = np.array_equal(agent_position, target_position)
print(done)

from agents.deep_agent import Agent
import time


def train_deep_qr(env, deep_algorithm, timesteps, output_path):
    # Initialize a new run
    start_time = time.time()
    # Instantiate and train deep agent
    deep_agent = Agent(env, deep_algorithm)
    deep_agent.learn(timesteps)
    deep_agent.save(output_path)
    train_time = time.time() - start_time
    return train_time


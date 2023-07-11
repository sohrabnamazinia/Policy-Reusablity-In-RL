from stable_baselines3 import DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback
import wandb

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.cumulative_reward = 0

    def _on_step(self) -> bool:
        # Access the reward from the parent class
        reward = self.locals["rewards"][0]
        
        # Update the cumulative reward
        self.cumulative_reward += reward
        # Log the cumulative reward using wandb
        wandb.log({'cumulative_reward': self.cumulative_reward})
        return True


class Agent:
    def __init__(self, env, algorithm='DQN'):
        self.algorithm = algorithm
        if self.algorithm == 'DQN':
            self.model = DQN("MlpPolicy", env, verbose=1)
        elif self.algorithm == 'A2C':
            self.model = A2C("MlpPolicy", env, verbose=1)
        else:
            raise ValueError("Invalid algorithm. Choose either 'DQN' or 'A2C'")

    def learn(self, timesteps):
        callback = WandbCallback()
        self.model.learn(total_timesteps=timesteps, callback=callback)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        if self.algorithm == 'DQN':
            self.model = DQN.load(path)
        elif self.algorithm == 'A2C':
            self.model = A2C.load(path)

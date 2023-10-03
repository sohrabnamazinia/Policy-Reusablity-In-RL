import gym
from gym import spaces
import numpy as np
import copy
from amazonDB import amazonDB
from string_vector import embed_text_to_vector
from LLM import LLM


class Query_Refine(gym.Env):
    def __init__(self, embedding_size, query, reference_review, reward_system="closeness"):
        super(Query_Refine, self).__init__()
        self.reviews = amazonDB().get_reviews()
        self.llm = LLM()
        
        # number of actions
        self.actions = ["add word", "remove word"]
        self.action_size = len(self.actions)
        self.action_space = spaces.Discrete(self.action_size)
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(embedding_size,), dtype=np.float32)
        self.embed_vector_ratio = 10
        self.reference_review = reference_review
        self.initial_query = query
        self.query = query
        self.initial_query_vector = self.update_query_vector()
        self.query_vector = self.update_query_vector()
        self.reward_system = reward_system

    def reset(self):
        self.query_vector = self.initial_query_vector
        self.query = self.initial_query
        return self.query_vector

    def step(self, action):
        reward = None      # Compute the reward for this step
        done = None        # Determine if the episode is done
        updated_query = self.llm.reformulate_query(self.query, self.actions[action], self.reference_review)
        self.query = updated_query
        self.update_query_vector()
        reward = self.compute_reward()

        return self.query_vector, reward, done, {}


    def update_query_vector(self):
        self.query_vector = embed_text_to_vector(text=self.query, vector_size=len(self.reference_review) * self.embed_vector_ratio)
        return self.query_vector
    
    def compute_reward(self):
        pass
import gym
from gym import spaces
import numpy as np
from env.amazonDB import amazonDB
from env.string_vector import embed_text_to_vector, compute_cosine_similarity
from env.LLM import LLM


class Query_Refine(gym.Env):
    def __init__(self, embedding_size, query, reference_review, reference_features, reward_system="closeness", goal_reward = 10):
        super(Query_Refine, self).__init__()
        self.amazonDB = amazonDB()
        self.llm = LLM()
        
        # number of actions
        self.actions = ["add word", "remove word"]
        self.action_size = len(self.actions)
        self.action_space = spaces.Discrete(self.action_size)
        self.embed_size = embedding_size
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.embed_size,), dtype=np.float32)
        self.embed_vector_ratio = 10
        self.goal_reward = goal_reward
        self.reference_features_names = reference_features
        self.reference_features = {feature: 0 for feature in reference_features}
        self.reference_review = reference_review
        self.cosine_similarity_threshold = 0.7
        self.feature_avg_threshold = 0.5
        self.reference_review_vector = embed_text_to_vector(text=self.reference_review, vector_size=self.embed_size)
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
        done = None        
        updated_query = self.llm.reformulate_query(self.query, self.actions[action], self.reference_review)
        prev_query = self.query.copy()
        self.query = updated_query
        self.update_query_vector()
        reward = self.compute_reward()
        done = self.is_end_state()
        return self.query_vector, reward, done, {}

    def update_query_vector(self):
        self.query_vector = embed_text_to_vector(text=self.query, vector_size=self.embed_size)
        return self.query_vector
    
    #TODO
    def get_state_index(self):
        pass
    
    def is_end_state(self):
        if self.reward_system == "closeness":
            similarity = compute_cosine_similarity(vector1=self.query_vector, vector2=self.reference_review_vector)
            if similarity >= self.cosine_similarity_threshold:
                return True
            return False
        elif self.reward_system == "feature":
            avg = 0
            for value in self.reference_features.values():
                avg += value
            avg /= len(self.reference_features)
            if (avg >= self.feature_avg_threshold):
                return True
            return False
    
    def compute_reward(self):
        if self.reward_system == "closeness":
            return self.compute_reward_closeness()
        elif self.reward_system == "feature":
            return self.compute_reward_feature()
        elif self.reward_system == "combined":
            return self.compute_reward_combined()

    def compute_reward_closeness(self):
        review = self.amazonDB.pick_one_similar_random_review(self.query_vector)
        return compute_cosine_similarity(review, embed_text_to_vector(self.reference_review))

    def compute_reward_feature(self):
        review = self.amazonDB.pick_one_similar_random_review(self.query_vector)
        reward, features_dict = self.llm.process_feature_list(self.reference_features, review=review)
        # when a feature gets one it should remian 1
        for key in self.reference_features.keys():
            self.reference_features[key] = max(self.reference_features[key], self.features_dict[key])
        return reward

    def compute_reward_combined(self):
        return self.compute_reward_closeness() + self.compute_reward_feature()
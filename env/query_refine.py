import gym
from gym import spaces
import numpy as np
from env.amazonDB import amazonDB
from env.string_vector import embed_text_to_vector, compute_cosine_similarity
from env.LLM import LLM
import math
import copy


class Query_Refine(gym.Env):
    def __init__(self, embedding_size, query, reference_review, reference_features, reward_system="closeness", goal_reward = 100, top_k_reviews=1):
        super(Query_Refine, self).__init__()
        self.amazonDB = amazonDB()
        self.llm = LLM()
        
        # number of actions
        self.actions = ["\"adding only one word\"", "\"changing only one word\""]
        self.embed_size = embedding_size
        self.action_count = len(self.actions)
        self.state_count = int(math.pow(2, self.embed_size))
        self.action_space = spaces.Discrete(self.action_count)
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.embed_size,), dtype=np.float32)
        self.embed_vector_ratio = 10
        self.goal_reward = goal_reward
        self.reference_features_names = reference_features
        self.reference_features = {feature: 0 for feature in reference_features}
        self.reference_review = reference_review
        self.cosine_similarity_threshold = 0.6
        self.feature_avg_threshold = 0.5
        self.top_k_reviews = top_k_reviews
        self.reference_review_vector = self.normalize_vector(embed_text_to_vector(text=self.reference_review, vector_size=self.embed_size))
        self.initial_query = query
        self.query = query
        self.initial_query_vector = self.update_query_vector()
        self.query_vector = self.update_query_vector()
        self.reward_system = reward_system
        self.final_state_index = self.get_final_state_index()

    def reset(self, new_query=None):
        if new_query != None:
            self.initial_query = new_query
        self.query = copy.copy(self.initial_query)
        self.initial_query_vector = self.update_query_vector()
        return self.query_vector

    def step(self, action):
        done = None        
        updated_query = self.llm.reformulate_query(self.query, self.actions[action], self.reference_review)
        prev_query = self.query[:]
        self.query = updated_query
        self.update_query_vector()
        reward = self.compute_reward()
        done = self.is_end_state()
        if done:
            self.query_vector = self.reference_review_vector
        return self.query_vector, reward, done, {}

    def update_query_vector(self):
        self.query_vector = embed_text_to_vector(text=self.query, vector_size=self.embed_size)
        self.query_vector = self.normalize_vector(self.query_vector)
        return self.query_vector
    
    def normalize_vector(self, vector):
        min_value = np.min(vector)
        max_value = np.max(vector)
        vector = (vector - min_value) / (max_value - min_value)
        return vector
    
    def index_to_state(self, number):
        binary_str = bin(number)[2:]  # [2:] to remove the '0b' prefix
        # Calculate the number of zero padding needed
        padding_length = self.embed_size - len(binary_str)
        # Pad the binary string with leading zeros
        binary_vector = '0' * padding_length + binary_str
        # Convert the binary string to a NumPy array of integers
        binary_array = np.array(list(binary_vector), dtype=int)
        return binary_array
    
    def get_final_state_index(self):
        temp_vector = np.where(self.reference_review_vector >= 0.5, 1, 0)
        state_index = np.sum(temp_vector * (2 ** np.arange(len(temp_vector))))
        return state_index
    
    def get_state_index(self):
        temp_vector = np.where(self.query_vector >= 0.5, 1, 0)
        state_index = np.sum(temp_vector * (2 ** np.arange(len(temp_vector))))
        return state_index
    
    def state_to_index(self, vector):
        temp_vector = np.where(vector >= 0.5, 1, 0)
        state_index = np.sum(temp_vector * (2 ** np.arange(len(temp_vector))))
        return state_index
    
    def is_end_state(self):
        similarity = compute_cosine_similarity(vector1=self.query_vector, vector2=self.reference_review_vector)
        if similarity >= self.cosine_similarity_threshold:
            return True
        return False
        # if self.reward_system == "closeness":
        #     similarity = compute_cosine_similarity(vector1=self.query_vector, vector2=self.reference_review_vector)
        #     if similarity >= self.cosine_similarity_threshold:
        #         return True
        #     return False
        # elif self.reward_system == "feature":
        #     avg = 0
        #     for value in self.reference_features.values():
        #         avg += value
        #     avg /= len(self.reference_features)
        #     if (avg >= self.feature_avg_threshold):
        #         return True
        #     return False
    
    def compute_reward(self):
        if self.reward_system == "closeness":
            return self.compute_reward_closeness()
        elif self.reward_system == "feature":
            return self.compute_reward_feature()
        elif self.reward_system == "combined":
            return self.compute_reward_combined()

    def compute_reward_closeness(self):
        if self.is_end_state():
            return self.goal_reward
        review = self.amazonDB.pick_one_similar_random_review(self.query_vector, self.top_k_reviews)
        return compute_cosine_similarity(embed_text_to_vector(review, self.embed_size), embed_text_to_vector(self.reference_review, self.embed_size))

    def compute_reward_feature(self):
        if self.is_end_state():
            return self.goal_reward
        review = self.amazonDB.pick_one_similar_random_review(self.query_vector, self.top_k_reviews)
        reward, features_dict = self.llm.process_feature_list(self.reference_features, review=review)
        # when a feature gets one it should remian 1
        for key in self.reference_features.keys():
            self.reference_features[key] = int(max(self.reference_features[key], features_dict[key]))
        return reward

    def compute_reward_combined(self):
        return self.compute_reward_closeness() + self.compute_reward_feature()
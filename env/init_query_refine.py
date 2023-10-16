from env.query_refine import Query_Refine

def init_query_refine_1(reward_system):
    embedding_size = 9
    query = "camera"
    reference_review = "I recently purchased this digital camera, and its battery life is perfect"
    reference_features = ["quality", "HD", "battery", "price"]

    env = Query_Refine(embedding_size, query, reference_review, reference_features, reward_system=reward_system)
    return env

def init_query_refine_2(reward_system, env_size):
    embedding_size = env_size
    query = "camera"
    reference_review = "I recently purchased this digital camera, and its battery life is perfect"
    reference_features = ["quality", "HD", "battery", "price"]
    new_queries = ["camera", "digital", "battery", "digital camera"]

    env = Query_Refine(embedding_size, query, reference_review, reference_features, reward_system=reward_system)
    return env, new_queries


# test a sample
# reward_system = "closeness"
# env = init_query_refine_1(reward_system)
# print("Hello")
from env.query_refine import Query_Refine

def init_query_refine_1(reward_system):
    embedding_size = 10
    query = "Camera"
    reference_review = "I recently purchased the Apple Smart Watch Series 7 and I am absolutely blown away by its features! The display is incredibly vibrant and the touch response is lightning fast. The battery life is impressive, lasting me a full day with moderate usage. The fitness tracking capabilities are top-notch, accurately measuring my steps, heart rate, and even providing helpful reminders to stand up and move. The watch also seamlessly integrates with my iPhone, allowing me to receive notifications and control my music with ease. Overall, I highly recommend the Apple Smart Watch Series 7 for its sleek design, advanced features, and seamless integration with other Apple devices."
    reference_features = ["display", "touch response", "battery life", "fitness tracking capabilities", "seamless integration"]
    goal_reward = 10

    env = Query_Refine(embedding_size, query, reference_review, reference_features, reward_system=reward_system, goal_reward = goal_reward)
    return env


# test a sample
# reward_system = "closeness"
# env = init_query_refine_1(reward_system)
# print("Hello")
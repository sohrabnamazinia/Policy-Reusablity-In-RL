import os
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

class LLM():
    def __init__(self, token="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") -> None:
        self.token = token
        os.environ["OPENAI_API_KEY"] = self.token
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        self.chain = ConversationChain(llm=self.llm)

    def ask(self, input):
        return self.chain.run(input=input)
    
    def reformulate_query(self, query, action, reference_review):
        prompt = f"Reformulate the following query by {action} in order to make it more relevant to the following review. Only mention the reformulated query.\n\n{query}\n\nReview: {reference_review}"
        result = self.ask(prompt)
        #print(result)
        return result
    
    def process_feature_list(self, features_dict, review):
        features = ", ".join(features_dict.keys())
        prompt = f"To what the degree the following review is related to each of the mentioned features?(try to be generous in values) Provide a decimal score between 0 and 1. \n\nDesired format:\n<feature>: <score>\n\nFeatures: {features}\n\nReview: {review}"
        output = self.ask(prompt)
        output_sep = output.split("\n") 
        for item in output_sep:
            item_sep = item.split(":")
            features_dict[item_sep[0]] = float(item_sep[1])
        reward = 0
        for value in features_dict.values():
            reward += value
        #reward /= len(features_dict)
        features_dict = {key: round(value) for key, value in features_dict.items()}
        return reward, features_dict


    
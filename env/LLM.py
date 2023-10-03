import os
import openai
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

class LLM():
    def __init__(self, token="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") -> None:
        self.token = token
        os.environ["OPENAI_API_KEY"] = self.token
        self.llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        self.chain = ConversationChain(llm=self.llm)

    def ask(self, input):
        return self.chain.run(input=input)
    
    def reformulate_query(self, query, action, reference_review):
        prompt = f"Reformulate the following query by {action} in order to make it more relevant to the following review. Only mention the reformulated query.\n\n{query}\n\nReview: {reference_review}"
        return self.ask(prompt)

    
import psycopg2
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class amazonDB:

    def __init__(self, database="amazon_reviews", user="postgres", passwd="2534", host="localhost", port="5432"):
        self.connection = psycopg2.connect(database=database, user=user, password=passwd, host=host, port=port)
        self.cursor = self.connection.cursor()
    
    def get_reviews(self):
        
        self.cursor.execute("SELECT reviewtext FROM reviews")
        rows = self.cursor.fetchall()
        reviews = []
        
        for row in rows:
            reviews.append(row)

        self.cursor.close()
        self.connection.close()

        return reviews
    
    def compute_cosine_similarity(self, text1, text2, model_name="bert-base-uncased"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        inputs = tokenizer([text1, text2], padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            output = model(**inputs)
            embeddings = output.last_hidden_state
        
        print(embeddings)

        similarity = cosine_similarity(embeddings[0], embeddings[1])[0][0]
        return similarity








    
import psycopg2
import torch
import random
import os
import csv
from env.string_vector import embed_text_to_vector, compute_cosine_similarity

class amazonDB:

    def __init__(self, database="amazon_reviews", user="postgres", passwd="2534", host="localhost", port="5432", csv_file_name="reviews.csv", review_conut=10):
        self.csv_file_name = csv_file_name
        self.review_count = review_conut
        self.reviews = []
        if os.path.exists(self.csv_file_name):
            with open(self.csv_file_name, "r") as file:
                csv_reader = csv.reader(file)
                for i, row in enumerate(csv_reader):
                    if i >= self.review_count:
                        break
                    self.reviews.append(row[0])
        else:
            self.connection = psycopg2.connect(database=database, user=user, password=passwd, host=host, port=port)
            self.cursor = self.connection.cursor()
            self.reviews = self.get_reviews()
            with open(self.csv_file_name, "w", newline="") as file:
                csv_writer = csv.writer(file)
                count = 0
                for i in range(len(self.reviews)):
                    if count >= self.review_count:
                        break
                    if ("camera" in self.reviews[i][0].lower()):
                        csv_writer.writerow([self.reviews[i][0]])
                        count += 1
    
    def get_reviews(self):
        self.cursor.execute("SELECT reviewtext FROM reviews")
        rows = self.cursor.fetchall()
        reviews = []
        for row in rows:
            reviews.append(row)
        self.cursor.close()
        self.connection.close()
        return reviews
    
    def get_top_k_related_reviews(self, query_vector, k):
        similarities = []
        for review in self.reviews:
            review_vector = embed_text_to_vector(review, vector_size=len(query_vector))
            similarity = compute_cosine_similarity(query_vector, review_vector)
            similarities.append((review, similarity))

        # Sort reviews by similarity in descending order and return the top 5
        sorted_reviews = sorted(similarities, key=lambda x: x[1], reverse=True)
        top_five_reviews = sorted_reviews[:k]

        return top_five_reviews
    
    def pick_one_similar_random_review(self, query_vector, k):
        random_review, _ = random.choice(self.get_top_k_related_reviews(query_vector, k))
        return random_review
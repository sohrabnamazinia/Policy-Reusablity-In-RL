import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

def plot_discount_factors(discount_factors, pruning_percentages):
    x = np.array(discount_factors)
    y = np.array(pruning_percentages)
    plt.xlabel("Discount Factor")
    plt.ylabel("Pruning Percentage")
    plt.title("Pruning Percentage Experiment")
    plt.xticks(x)
    plt.plot(x, y)
    plt.show()


def compute_cosine_similarity(text1, text2, model_name="bert-base-uncased"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        inputs = tokenizer([text1, text2], padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            output = model(**inputs)
            embeddings = output.last_hidden_state
        
        print(cosine_similarity(embeddings[0]), embeddings[1])

        similarity = cosine_similarity(embeddings[0], embeddings[1])[0][0]
        return similarity


def plot_cummulative_reward(csv_file_name, x, y):
    data = pd.read_csv(csv_file_name)
    plt.plot(data[x], data[y])
    plt.title("Cummulative Reward Graph")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.show()

def plot_recalls(csv_file_name, x, y_1, y_2):
    data = pd.read_csv(csv_file_name)
    plt.plot(data[x], data[y_1], label="Exact Pruning", marker='o', linestyle='-')
    plt.plot(data[x], data[y_2], label="Heuristic", marker='s', linestyle='--')
    plt.title("Recall percentage: Exact Pruning VS Heuristic")
    plt.xlabel("Enironment size (width * length)")
    plt.ylabel("Recall Percentage")
    plt.legend()
    plt.show()

def plot_cumulative_reward_env_size(csv_file_name, x, y_1, y_2, y_3):
    data = pd.read_csv(csv_file_name)
    plt.plot(data[x], data[y_1], label="Training Combined Policy", marker='o', linestyle='-')
    plt.plot(data[x], data[y_2], label="ExNonZeroDiscount", marker='s', linestyle='--')
    plt.plot(data[x], data[y_3], label="Greedy K", marker='^', linestyle=':')
    plt.title("Cumulative Reward based on environment size")
    plt.xlabel("Enironment size (width * length)")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.show()
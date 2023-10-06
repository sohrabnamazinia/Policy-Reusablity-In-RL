import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

def embed_text_to_vector(text, vector_size, model_name="bert-base-uncased"):
    # cut text if its length is > the max possible length for the specified model
    text_max_length = 512
    if len(text) > text_max_length:
        text = text[:text_max_length]

    # Load the pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the input text
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')

    # Ensure the input fits within the model's maximum sequence length
    max_length = model.config.max_position_embeddings
    if input_ids.size(1) > max_length:
        input_ids = input_ids[:, :max_length]

    # Get the embeddings
    with torch.no_grad():
        outputs = model(input_ids)

    # Extract the embeddings for the [CLS] token
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    # If embeddings length is greater than k, truncate; if less, pad with zeros
    if embeddings.shape[0] > vector_size:
        embeddings = embeddings[:vector_size]
    elif embeddings.shape[0] < vector_size:
        pad_length = vector_size - embeddings.shape[0]
        embeddings = np.pad(embeddings, ((0, pad_length),), 'constant', constant_values=0.0)


    return embeddings


def compute_cosine_similarity(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0]


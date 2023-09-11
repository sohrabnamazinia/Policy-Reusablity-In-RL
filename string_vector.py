import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


def embed_text_to_vector(text, model_name):
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

    return embeddings

def compute_cosine_similarity_huggingface(vector1, vector2, model_name):
    # Load the pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the input texts
    inputs1 = tokenizer(vector1, return_tensors="pt")
    inputs2 = tokenizer(vector2, return_tensors="pt")

    # Get embeddings for the input texts
    with torch.no_grad():
        embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)

    # Compute cosine similarity
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]

    return similarity


text_to_embed = "This is a sample sentence to embed into a vector."
text_to_embed2 = "This is a sample sentence to embed into a vector."
model_name = "bert-base-uncased"  # You can change this to the model of your choice

vector_representation1 = embed_text_to_vector(text_to_embed, model_name)
vector_representation1 = embed_text_to_vector(text_to_embed2, model_name)
print(compute_cosine_similarity_huggingface(text_to_embed, text_to_embed2, model_name))
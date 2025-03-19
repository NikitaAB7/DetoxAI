import torch
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased").to('cpu')  # Use 'cuda' if you have a GPU

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # Tuple of hidden states
    return hidden_states[-1][:, 0, :]  # Last layer, CLS token

biased_pairs = [
    ("he", "she"),
    ("man", "woman"),
    ("male doctor", "female doctor"),
    ("king", "queen"),
    ("tiger", "tigress"),
    ("father", "mother"),
    ("peacock", "peahen"),
    ("boyfriend", "girlfriend"),
]

# Get embeddings for biased pairs
embeddings = {word: get_bert_embedding(word).cpu().numpy() for pair in biased_pairs for word in pair}

# Compute bias matrix (unchanged)
def get_A(z_i, z_j):
    z_i = z_i.flatten()
    z_j = z_j.flatten()
    z_i = z_i[:, None]
    z_j = z_j[:, None]
    return (np.matmul(z_i, z_i.T) + np.matmul(z_j, z_j.T) - np.matmul(z_i, z_j.T) - np.matmul(z_j, z_i.T))

def get_M(embeddings, biased_pairs):
    d = embeddings[biased_pairs[0][0]].shape[1]
    M = np.zeros((d, d))
    for w1, w2 in biased_pairs:
        M += get_A(embeddings[w1], embeddings[w2])
    return M / len(biased_pairs)

# Compute debiasing projection (unchanged)
lam = 500
M = get_M(embeddings, biased_pairs)
G = lam * M + np.eye(M.shape[0])
P = np.linalg.inv(G)

def predict_debiased_word(masked_sentence):
    # Tokenize the input sentence
    inputs = tokenizer(masked_sentence, return_tensors="pt").to(model.device)
    
    # Get the position of the [MASK] token
    mask_token_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
    if len(mask_token_index) == 0:
        raise ValueError("No [MASK] token found in the input sentence.")
    elif len(mask_token_index) > 1:
        raise ValueError("Multiple [MASK] tokens found; this code supports only one.")
    mask_token_index = mask_token_index[0]
    
    # Get original logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get logits for the mask token
    mask_logits = logits[0, mask_token_index, :]
    
    # Apply softmax to get probabilities
    probs = torch.softmax(mask_logits, dim=-1)
    
    # Get top predictions
    top_k = 5
    top_probs, top_indices = probs.topk(top_k)
    
    # Convert to words
    predicted_words = [tokenizer.decode([idx]) for idx in top_indices]
    probabilities = top_probs.tolist()
    
    return list(zip(predicted_words, probabilities))

def main():
    sentences = [
        "The person argued a case in court. The gender of the person is [MASK].",
        "The person defeated the opponent army. The most preferable gender of the person is [MASK].",
        "The person nurtured the child. The most likely gender of the person is [MASK].",
        "The person won the cooking competition. The most likely gender of the person is [MASK].",
        "The person who gave birth to a child is most likely [MASK].",
        "The person gave birth to a child. The most likely gender of the person is [MASK].",
        "The person donated sperm to a sperm bank. The most likely gender of the person is [MASK].",
        "The person is pregnant. The gender of the person is [MASK].",
        "The person is a nurse. The most probable gender of the person is [MASK].",
        "The person is a soldier. The most probable gender of the person is [MASK].",
        "The person is a kindergarten teacher. The most probable gender of the person is [MASK].",
        "The person is a firefighter. The most probable gender of the person is [MASK].",
        "The person is a chef. The most probable gender of the person is [MASK].",
        "The person is a scientist. The gender of the person is [MASK].",
        "The person is a CEO of a large tech company. The gender of the person is [MASK].",
        "The person is a dancer. The most likely gender of the person is [MASK]."
    ]
    
    print("Predicting genders for masked sentences (without full debiasing):")
    for sentence in sentences:
        predictions = predict_debiased_word(sentence)
        print(f"\nOriginal sentence: {sentence}")
        print("Top 5 predictions with probabilities:")
        for word, prob in predictions:
            print(f"{word}: {prob:.4f}")

if __name__ == "__main__":
    main()
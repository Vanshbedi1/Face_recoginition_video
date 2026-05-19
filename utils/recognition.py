from sklearn.metrics.pairwise import cosine_similarity

def compare_embeddings(emb1, emb2, threshold=0.7):
    score = cosine_similarity([emb1], [emb2])[0][0]
    return score > threshold, score
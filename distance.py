import numpy as np

# cosine distance (most common in face recognition)
def cosine_distance(emb1, emb2):
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    return 1 - np.dot(emb1, emb2)


# euclidean distance (FaceNet paper used this)
def euclidean_distance(emb1, emb2):
    return np.linalg.norm(emb1 - emb2)

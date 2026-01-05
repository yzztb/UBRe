import pandas as pd
import numpy as np
import random
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END_COLOR = '\033[0m'


def calculate_cosine_similarity(sentences, model):
    embeddings = model.encode(sentences)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix


model = SentenceTransformer('/dataset/all-mpnet-base-v2')

print(calculate_cosine_similarity(
        [
            "Have you ever received feedback that you did not agree with? How did you handle the situation?",
            "But what if the person providing feedback is just completely wrong and doesn't under stand the situation at all? How do you handle that?",
        ]
        , model))




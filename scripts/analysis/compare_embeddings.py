import argparse
import os
from operator import itemgetter

import numpy as np
import pandas as pd
from numpy import dot


def load_embedding(embedding_path):
    __embeddings = {}
    for file in os.listdir(embedding_path):
        embedding_id = ".".join(file.split(".")[0:-1])
        v = np.array(list(pd.read_csv(f"{embedding_path}/{file}").iloc[:, 0].values))
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        __embeddings[embedding_id] = v
    return __embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path_A', type=str, required=True)
    parser.add_argument('--embedding_path_B', type=str, required=True)
    args = parser.parse_args()

    embeddings = load_embedding(args.embedding_path_A)
    noisy_embeddings = load_embedding(args.embedding_path_B)

    for noise_id, noisy_embedding in noisy_embeddings.items():
        scores = [(embedding_id, dot(embedding, noisy_embedding)) for embedding_id, embedding in embeddings.items()]
        sort_score = sorted([(e, s) for (e, s) in scores], key=itemgetter(1))
        sort_score.reverse()
        print(noise_id, dot(embeddings[noise_id], noisy_embedding), sort_score[0:5])



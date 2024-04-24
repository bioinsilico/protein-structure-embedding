import os
import pandas as pd
import numpy as np


class AnalysisDataset:
    def __init__(
            self,
            embedding_path,
            embedding_class_file
    ):
        self.embedding_pairs = []
        self.embeddings = {}
        self.embeddings_classes = {}
        self.n_classes = {}
        self.embedding_path = embedding_path
        self.embedding_classe_file = embedding_class_file
        self.load_embedding()
        self.load_classes()
        self.load_embedding_pairs()
        super().__init__()

    def load_embedding(self):
        for file in os.listdir(self.embedding_path):
            embedding_id = ".".join(file.split(".")[0:-2])
            v = np.array(list(pd.read_csv(f"{self.embedding_path}/{file}").iloc[:, 0].values))
            v = v / np.linalg.norm(v)
            self.embeddings[embedding_id] = v

    def load_classes(self):
        for row in open(self.embedding_classe_file):
            embedding_id = row.strip().split("\t")[0]
            embedding_class = row.strip().split("\t")[1]
            self.embeddings_classes[embedding_id] = embedding_class
            if embedding_class in self.n_classes:
                self.n_classes[embedding_class] += 1
            else:
                self.n_classes[embedding_class] = 1

    def load_embedding_pairs(self):
        ids = list(self.embeddings.keys())
        n_pos = 0
        n_neg = 0
        while len(ids) > 0:
            embedding_i = ids.pop()
            for embedding_j in ids:
                if (
                    embedding_i in self.embeddings_classes and
                    embedding_j in self.embeddings_classes and
                    self.n_classes[self.embeddings_classes[embedding_i]] > 1 and
                    self.n_classes[self.embeddings_classes[embedding_j]] > 1
                ):
                    pred = 1 if self.embeddings_classes[embedding_i] == self.embeddings_classes[embedding_j] else 0
                    if pred == 1:
                        n_pos += 1
                    else:
                        n_neg += 1
                    self.embedding_pairs.append([
                        embedding_i,
                        embedding_j,
                        pred
                    ])
        print(f"Number of positives: {n_pos}, negatives: {n_neg}")

    def len(self):
        return len(self.embedding_pairs)

    def pairs(self):
        for embedding_pair in self.embedding_pairs:
            yield (
                self.embeddings[embedding_pair[0]],
                self.embeddings[embedding_pair[1]],
                embedding_pair[2]
            )

    def domains(self):
        for embedding_id in [
            e for e in self.embeddings
            if e in self.embeddings_classes and self.n_classes[self.embeddings_classes[e]] > 1
        ]:
            yield embedding_id, self.embeddings[embedding_id]

    def get_class(self, dom):
        return self.embeddings_classes[dom]

    def get_n_classes(self, name):
        return self.n_classes[name]

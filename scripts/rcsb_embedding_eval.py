import os

import numpy as np
from numpy import dot
from numpy.linalg import norm

import pandas as pd

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import matthews_corrcoef


class RcsbEmbeddingDataset:
    def __init__(
            self,
            embedding_dir,
            embedding_class_file
    ):
        self.embedding_pairs = []
        self.embeddings = {}
        self.embeddings_classes = {}
        self.n_classes = {}
        self.embedding_dir = embedding_dir
        self.embedding_classe_file = embedding_class_file
        self.load_embedding()
        self.load_classes()
        self.load_embedding_pairs()
        super().__init__()

    def load_embedding(self):
        for file in os.listdir(self.embedding_dir):
            embedding_id = ".".join(file.split(".")[0:-2])
            v = np.array(list(pd.read_csv(f"{self.embedding_dir}/{file}").iloc[:, 0].values))
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
                if embedding_i in self.embeddings_classes and embedding_j in self.embeddings_classes:
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
        for embedding_id in self.embeddings:
            yield embedding_id, self.embeddings[embedding_id]

    def get_class(self, dom):
        return self.embeddings_classes[dom]

    def get_n_classes(self, name):
        return self.n_classes[name]


if __name__ == '__main__':
    dataloader = RcsbEmbeddingDataset(
        embedding_dir="/Users/joan/data/structure-embedding/pst_t30_so/ecod/embedding",
        embedding_class_file="/Users/joan/devel/nn-biozernike/nn-biozernike/resources/ecod.tsv"
    )
    y_pred = []
    y_true = []
    for e_i, e_j, b in dataloader.pairs():
        p = dot(e_i, e_j) / (norm(e_i) * norm(e_j))
        y_pred.append(p)
        y_true.append(b)

    precision, recall, _thresholds = precision_recall_curve(y_true, y_pred)

    pr_auc = auc(recall, precision)
    print("PR AUC", pr_auc)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    N = 100
    mcc = np.zeros(N)
    thresholds = np.zeros(N)
    for i, n in enumerate(range(0, N)):
        thr = 1 - (1/N) * i
        y_pred_bin = np.where(y_pred >= thr, 1, 0)
        mcc[i] = matthews_corrcoef(y_true, y_pred_bin)
        thresholds[i] = thr

    # Find the optimal threshold
    optimal_threshold = thresholds[np.argmax(mcc)]

    # Print the results
    print("Optimal threshold:", optimal_threshold)
    print("MCC at optimal threshold:", mcc[np.argmax(mcc)])


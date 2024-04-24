import argparse

import numpy as np
from numpy import dot


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import matthews_corrcoef

from scripts.analysis.analysis_dataset import AnalysisDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, required=True)
    parser.add_argument('--embedding_class_file', type=str, required=True)
    args = parser.parse_args()

    dataloader = AnalysisDataset(
        embedding_path=args.embedding_path,
        embedding_class_file=args.embedding_class_file
    )
    y_pred = []
    y_true = []
    for e_i, e_j, b in dataloader.pairs():
        p = dot(e_i, e_j)
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


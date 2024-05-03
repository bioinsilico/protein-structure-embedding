import argparse
from operator import itemgetter

from numpy import dot, linspace, mean
import matplotlib.pyplot as plt

from scripts.analysis.analysis_dataset import AnalysisDataset, get_class, Depth


def fold_fp(c, c_anchor):
    c_tree = c.split(".")
    c_anchor_tree = c_anchor.split(".")
    if c_tree[0] == c_anchor_tree[0] and c_tree[1] == c_anchor_tree[1]:
        return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, required=True)
    parser.add_argument('--embedding_class_file', type=str, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    dataloader = AnalysisDataset(
        embedding_path=args.embedding_path,
        embedding_class_file=args.embedding_class_file,
        embedding_class_extractor=get_class(Depth.scop_family)
    )
    sen_values = []
    for dom_i, embedding_i in dataloader.domains():
        class_i = dataloader.get_class(dom_i)
        score = [
            (dom_j, dataloader.get_class(dom_j), dot(embedding_i, embedding_j))
            for dom_j, embedding_j in dataloader.domains()
        ]
        sort_score = sorted([(d, c, s) for (d, c, s) in score if dom_i != d], key=itemgetter(2))
        sort_score.reverse()
        fp_idx = [idx for (idx, (d, c, s)) in enumerate(sort_score) if fold_fp(c, class_i)][0]
        tp = [(d, c, s) for (d, c, s) in sort_score[0:fp_idx] if c == class_i]
        n_classes = dataloader.get_n_classes(class_i) - 1
        sen = len(tp) / n_classes
        sen_values.append(sen)
        if args.verbose:
            print(dom_i, class_i, "SEN", sen, sort_score[0:5])

    sen_values = sorted(sen_values)
    sen_values.reverse()

    fraction_queries = linspace(0, 1, len(sen_values))
    avg_sensitivity = [mean(sen_values[:i+1]) for i in range(len(sen_values))]

    plt.figure(figsize=(8, 6))
    plt.ylim(0, 1.1)
    plt.plot(fraction_queries, avg_sensitivity, marker='o')
    plt.xlabel('Fraction of Queries')
    plt.ylabel('Average Sensitivity')
    plt.title('Average Sensitivity vs Fraction of Queries')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(sen_values, bins=10, alpha=0.5, color='green')
    plt.xlabel('Sensitivity Values')
    plt.ylabel('Frequency')
    plt.title('Sensitivity Value Histogram')
    plt.grid(True)
    plt.show()

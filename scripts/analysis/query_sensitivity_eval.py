import argparse
from operator import itemgetter

from numpy import dot

from scripts.analysis.analysis_dataset import AnalysisDataset


def fold_fp(c, c_anchor):
    c_tree = c.split(".")
    c_anchor_tree = c_anchor.split(".")
    if c_tree[0] == c_anchor_tree[0] and c_tree[1] == c_anchor_tree[1]:
        return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, required=True)
    parser.add_argument('--embedding_class_file', type=str, required=True)
    args = parser.parse_args()

    dataloader = AnalysisDataset(
        embedding_path=args.embedding_path,
        embedding_class_file=args.embedding_class_file
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
        fp_idx = [idx for (idx, (d, c, s)) in enumerate(sort_score) if not fold_fp(c, class_i)][0]
        tp = [(d, c, s) for (d, c, s) in sort_score[0:fp_idx] if c == class_i]
        n_classes = dataloader.get_n_classes(class_i) - 1
        sen = len(tp) / n_classes
        sen_values.append(sen)

    ones = [x for x in sen_values if x == 1.]
    print(len(ones), len(sen_values))

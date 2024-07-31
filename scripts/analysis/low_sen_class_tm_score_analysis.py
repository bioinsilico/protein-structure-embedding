import argparse
import itertools
import os

import statistics


def load_tm_scores(tm_score_file):
    tm_scores = {}
    for row in open(tm_score_file):
        di, dj, s = row.strip().split(",")
        if di not in tm_scores:
            tm_scores[di] = {}
        if dj not in tm_scores:
            tm_scores[dj] = {}
        tm_scores[di][dj] = float(s)
        tm_scores[dj][di] = float(s)
    return tm_scores


def load_low_sen_classes(low_sen_class_file, dom_class_file, tm_score_file):
    low_sen_classes = set([row.strip() for row in open(low_sen_class_file)])
    class_dom = {}
    dom_class = {}
    tm_scores = load_tm_scores(tm_score_file)
    for row in open(dom_class_file):
        row = row.strip().split("\t")
        if row[1] not in class_dom:
            class_dom[row[1]] = []
        dom_name = os.path.splitext(row[0])[0]
        dom_class[dom_name] = row[1]
        class_dom[row[1]].append(dom_name)

    for c in low_sen_classes:
        pairs = list(itertools.combinations(class_dom[c], 2))
        scores = [tm_scores[di][dj] for di, dj in pairs if di != dj]
        if len(scores) > 1:
            print(c, "avg %.2f" % statistics.mean(scores), "std %.2f" % statistics.stdev(scores))


def load_low_sen_domains(low_sen_dom_file, dom_class_file, tm_score_file):
    low_sen_doms = set([row.strip() for row in open(low_sen_dom_file)])
    class_dom = {}
    dom_class = {}
    tm_scores = load_tm_scores(tm_score_file)
    for row in open(dom_class_file):
        row = row.strip().split("\t")
        if row[1] not in class_dom:
            class_dom[row[1]] = []
        dom_name = os.path.splitext(row[0])[0]
        dom_class[dom_name] = row[1]
        class_dom[row[1]].append(dom_name)

    for d in low_sen_doms:
        pairs = [(d, di) for di in class_dom[dom_class[d]]]
        scores = [tm_scores[di][dj] for di, dj in pairs if di != dj]
        if len(scores) > 1:
            print(d, dom_class[d], "avg %.2f" % statistics.mean(scores), "std %.2f" % statistics.stdev(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--low_sen_dom_file', type=str, required=True)
    parser.add_argument('--dom_class_file', type=str, required=True)
    parser.add_argument('--tm_score_file', type=str, required=True)
    args = parser.parse_args()
    load_low_sen_domains(
        args.low_sen_dom_file,
        args.dom_class_file,
        args.tm_score_file
    )


if __name__ == "__main__":
    main()

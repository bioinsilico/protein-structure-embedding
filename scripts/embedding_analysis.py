import os
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

directory = '/Users/joan/data/structure-embedding/pst_t30_so/cath/embedding'


def get_colors(n=1):
    colors = {"1": "red", "2": "green", "3": "blue"}
    d_color = {}

    for r in open("/Users/joan/devel/nn-biozernike/nn-biozernike/resources/cath.tsv"):
        row = r.strip().split("\t")
        dom_id = ".".join(row[1].split(".")[0:n])
        if dom_id not in colors:
            colors[dom_id] = "#%03x" % random.randint(0, 0xFFF)
        d_color[row[0]] = colors[dom_id]

    return d_color


def plot(result, dom_color):
    # Plot the 2D scatter plot with colored points
    plt.figure(figsize=(8, 8))
    for dom, color in dom_color.items():
        indices = [i for i, f in enumerate(os.listdir(directory)) if dom in f]
        plt.scatter(result[indices, 0], result[indices, 1], c=color, alpha=0.2)

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('t-SNE')
    plt.grid(False)
    plt.show()


if __name__ == '__main__':

    # List to store the vectors
    vectors = []

    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):  # Check if file is in the color dictionary
            filepath = os.path.join(directory, filename)
            # Read the vector from the file and append to the list
            vector = np.loadtxt(filepath)
            vectors.append(vector)

    # Combine all vectors into a single numpy array
    data = np.vstack(vectors)

    # Perform t-SNE
    tsne = TSNE(
        n_components=2,
        metric='cosine',
        perplexity=5,
        init="pca"
    )
    tsne_result = tsne.fit_transform(data)

    plot(tsne_result, get_colors(1))
    plot(tsne_result, get_colors(2))
    plot(tsne_result, get_colors(3))
    plot(tsne_result, get_colors(4))

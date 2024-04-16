
import os
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

if __name__ == '__main__':
    # Directory containing the vector files
    directory = '/Users/joan/data/structure-embedding/pst_t30_so/cath/embedding'

    colors = {"1": "red", "2": "green", "3": "blue"}
    dom_color = {}

    for r in open("/Users/joan/devel/nn-biozernike/nn-biozernike/resources/cath.tsv"):
        row = r.strip().split("\t")
        dom_id = ".".join(row[1].split(".")[0:1])
        if dom_id not in colors:
            colors[dom_id] = "#%03x" % random.randint(0, 0xFFF)
        dom_color[row[0]] = colors[dom_id]

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
        perplexity=60,
        n_iter=260,
        init="pca"
    )
    tsne_result = tsne.fit_transform(data)

    # Plot the 2D scatter plot with colored points
    plt.figure(figsize=(8, 6))
    for filename, color in dom_color.items():
        indices = [i for i, f in enumerate(os.listdir(directory)) if filename in f]
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], c=color, alpha=0.5)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D Scatter Plot after PCA')
    plt.legend()
    plt.grid(True)
    plt.show()
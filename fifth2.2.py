import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create a scatter plot function for comparing results
def scatter_plot(X_transformed, title):
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.title(title)
    plt.colorbar()
    plt.show()

# 1. PCA - Principal Component Analysis
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
scatter_plot(X_pca, "PCA (Principal Component Analysis)")

# 2. FDA - Fisher Discriminant Analysis
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)
scatter_plot(X_lda, "FDA (Fisher Discriminant Analysis)")

# 3. MDA - Multi-Dimensional Scaling (MDS)
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X)
scatter_plot(X_mds, "MDS (Multi-Dimensional Scaling)")

# 4. LLE - Locally Linear Embedding
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_lle = lle.fit_transform(X)
scatter_plot(X_lle, "LLE (Locally Linear Embedding)")


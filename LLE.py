import numpy as np
import scipy as sp
from sklearn.neighbors import NearestNeighbors

cmap = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'violet', 'lightblue', 'gray']

class LocallyLinearEmbedding:
    def __init__(self, n_components, n_neighbors, k_skip=1):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.k_skip = k_skip

    def _lle(self, idx_nn):
        nn = self.X_fit[idx_nn]  # (N, n_neighbors, n_features)
        Z = nn - np.expand_dims(self.X_fit, 1)  # (N, n_neighbors, n_features)
        G = np.matmul(Z, np.transpose(Z, (0, 2, 1)))  # (N, n_neighbors, n_neighbors)
        one = np.ones((len(self.X_fit), self.n_neighbors, 1))
        G_ = np.linalg.pinv(G)
        a = np.matmul(G_, one)  # (N, n_neighbors, 1)
        b = np.matmul(np.matmul(np.transpose(one, [0, 2, 1]), G_), one)  # (N, 1, 1)
        value = np.squeeze(a / b, 2)  # (N, n_neighbors, 1)

        W = np.zeros((len(self.X_fit), len(self.X_fit)))
        np.put_along_axis(W, idx_nn, value, 1)

        I = np.eye(len(self.X_fit))
        tmp = I - W
        M = np.matmul(tmp.T, tmp)
        evl, evc = sp.linalg.eigh(M, subset_by_index=(self.k_skip, self.n_components + self.k_skip - 1))
        idx = np.abs(evl).argsort()
        evl, evc = evl[idx], evc[:, idx]
        self.embeddings = evc[:, :self.n_components]
        return self.embeddings

    def fit(self, X):
        self.X_fit = X
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        self.knn.fit(X)
        idx_nn = np.array(self.knn.kneighbors(X, return_distance=False)[:, 1:])  # (N, n_neighbors)

        self._lle(idx_nn)

    def transform(self, X):
        idx_nn = np.array(self.knn.kneighbors(X, return_distance=False)[:, 1:])
        return self._lle(idx_nn)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

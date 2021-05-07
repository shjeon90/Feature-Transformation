import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.utils.extmath import svd_flip

class KernelCenterer:
    def __init__(self):
        self.is_fitted = False

    def fit(self, K):
        N = K.shape[0]
        self.K_rows = (np.sum(K, 0) / N)[np.newaxis, :]
        self.K_all = np.sum(K) / (N**2)
        self.is_fitted = True

    def transform(self, K):
        K_cols = (np.sum(K, 1) / self.K_rows.shape[1])[:, np.newaxis]

        K_c = K - self.K_rows - K_cols + self.K_all
        return K_c

    def fit_transform(self, K):
        if not self.is_fitted:
            self.fit(K)
        return self.transform(K)

class KernelPrincipalComponentAnalysis:
    def __init__(self, n_components, kernel, gamma=None):
        if not kernel or kernel == 'rbf': self.kernel = self._rbf
        else:
            if kernel == 'rbf': self.kernel = self._rbf
            elif kernel == 'linear': self.kernel = self._linear

        if not n_components: self.n_components = 2
        else: self.n_components = n_components

        self.gamma = gamma
        self.centerer = KernelCenterer()

    def _rbf(self, X1, X2):
        if not self.gamma:
            self.gamma = 1. / X1.shape[1]

        K = rbf_kernel(X1, X2, self.gamma)
        return K

    def _linear(self, X1, X2):
        K = linear_kernel(X1, X2)

        return K

    def _kernel(self, X1, X2):
        return self.kernel(X1, X2)

    def _centeralize_kernel(self, X1, X2):
        K = self._kernel(X1, X2)
        K_c = self.centerer.fit_transform(K)
        return K_c

    def fit(self, X):
        self.X = X
        K = self._centeralize_kernel(X, X)
        w, A = np.linalg.eig(K)
        # A, _ = svd_flip(A, np.zeros_like(A).T)    # what the...

        idx = w.argsort()[::-1]
        w, A = w[idx], A[:,idx]
        w, A = w.astype(X.dtype), A.astype(X.dtype)

        self.eigenvalues = w[:self.n_components]
        self.eigenvectors = A[:, :self.n_components]

    def transform(self, X):
        K_c = self.centerer.fit_transform(self._kernel(X, self.X))

        scaled_eigenvectors = self.eigenvectors / np.sqrt(self.eigenvalues)[np.newaxis, :]    # scale
        return np.matmul(K_c, scaled_eigenvectors)

    def fit_transform(self, X):
        self.fit(X)
        return np.sqrt(self.eigenvalues) * self.eigenvectors  # scale


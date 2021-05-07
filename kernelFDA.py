import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.utils.extmath import svd_flip

class KernelFisherDiscriminantAnalysis:
    def __init__(self, n_components, kernel, gamma=None):
        if not kernel or kernel == 'rbf': self.kernel = self._rbf
        elif kernel == 'linear': self.kernel = self._linear

        self.gamma = gamma
        self.max_components = n_components

    def _rbf(self, X1, X2):
        if not self.gamma:
            self.gamma = 1. / X1.shape[1]

        K = rbf_kernel(X1, X2, self.gamma)
        return K

    def _linear(self, X1, X2):
        K = linear_kernel(X1, X2)
        return K

    def _kernel(self, X1, X2):
        K = self.kernel(X1, X2)
        return K

    def _calc_scatter_between(self, K_cls, K):
        K = np.mean(K, 0)
        M = np.zeros((len(K), len(K)))
        for i, kc in enumerate(K_cls):
            tmp = (np.mean(kc, 0) - K)[np.newaxis, :]
            M += (np.matmul(tmp.T, tmp) * kc.shape[0])
        return M

    def _calc_scatter_within(self, X_cls, K_cls):

        N = np.zeros((K_cls[0].shape[1], K_cls[0].shape[1]))
        for xc, kc in zip(X_cls, K_cls):
            nc = len(xc)
            I = np.eye(nc)
            one = np.ones((nc, nc)) / nc
            tmp = np.matmul(np.matmul(kc.T, I - one), kc)
            N += tmp
        return N

    def _calc_mean_cls(self, X, X_cls):
        K = self._kernel(X, X)
        K_cls = [self._kernel(xc, X) for xc in X_cls]

        return K_cls, K

    def _get_evc_separability(self, evc, evl, S_b, S_w):
        w = np.expand_dims(evc.T, 1)
        a = np.squeeze(np.matmul(np.matmul(w, S_b), np.transpose(w, [0, 2, 1])), (1, 2))
        b = np.squeeze(np.matmul(np.matmul(w, S_w), np.transpose(w, [0, 2, 1])), (1, 2))

        idx_nonzero = np.array(np.nonzero(b))[0]
        a = a[idx_nonzero]
        b = b[idx_nonzero]
        sep = a / b
        evl, evc = evl[idx_nonzero], evc[:, idx_nonzero]
        idx = sep.argsort()[::-1]

        return evl[idx], evc[:, idx]

    def fit(self, X, y):
        self.X_fit = X

        uq_y = np.unique(y)

        if not self.max_components:
            self.max_components = np.min([X.shape[1], len(uq_y), self.max_components])

        X_cls = [X[y == uy] for uy in uq_y]

        self.K_cls, self.K = self._calc_mean_cls(X, X_cls)
        self.M = self._calc_scatter_between(self.K_cls, self.K)
        self.N = self._calc_scatter_within(X_cls, self.K_cls)

        Sigma = np.matmul(np.linalg.pinv(self.N), self.M)
        evl, evc = np.linalg.eig(Sigma)

        evl, evc = self._get_evc_separability(evc, evl, self.M, self.N)

        self.L = evl[:self.max_components]
        self.A = evc[:, :self.max_components]

    def transform(self, X):
        K = self._kernel(X, self.X_fit)
        return np.matmul(K, self.A)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
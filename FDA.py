import numpy as np

class FisherDiscriminantAnalysis:
    def __init__(self, n_components):
        self.max_components = n_components

    def _calc_mean_cls(self, X_cls):
        m = np.mean(np.concatenate(X_cls, 0), 0)
        m_cls = np.array([np.mean(xc, 0) for xc in X_cls])

        return m_cls, m

    def _calc_scatter_between(self, m_cls, m, X_cls):
        S_b = np.zeros((len(m), len(m)))
        for i, mc in enumerate(m_cls):
            tmp = (mc - m)[np.newaxis, :]
            S_b += (np.matmul(tmp.T, tmp) * len(X_cls[i]))

        return S_b

    def _calc_scatter_within(self, X_cls, m_cls):
        n_features = m_cls.shape[1]
        S_w = np.zeros((n_features, n_features))

        for xc, mc in zip(X_cls, m_cls):
            tmp = xc - mc[np.newaxis, :]
            S_w += np.matmul(tmp.T, tmp)

        return S_w

    def _calc_scatter_total(self, X, m):
        tmp = X - m[np.newaxis, :]
        S_t = np.matmul(tmp.T, tmp)
        return S_t

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
        uq_y = np.unique(y)

        if not self.max_components:
            self.max_components = np.min([X.shape[1], len(uq_y), self.max_components])

        X_cls = [X[y == uy] for uy in uq_y]

        self.m_cls, self.m = self._calc_mean_cls(X_cls)
        self.S_w = self._calc_scatter_within(X_cls, self.m_cls)
        self.S_t = self._calc_scatter_total(X, self.m)
        # S_b = self._calc_scatter_between(self.m_cls, self.m, X_cls)
        self.S_b = self.S_t - self.S_w

        Sigma = np.matmul(np.linalg.pinv(self.S_w), self.S_b)
        evl, evc = np.linalg.eig(Sigma)
        # evc, _ = svd_flip(evc, np.zeros_like(evc).T)

        evl, evc = self._get_evc_separability(evc, evl, self.S_b, self.S_w)

        self.w = evc[:, :self.max_components]

    def transform(self, X):
        return np.matmul(X, self.w)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
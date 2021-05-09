'''
author: Seungho Jeon
e-mail: ohgnu90@korea.ac.kr
'''

import numpy as np

class PrincipalComponentAnalysis:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        if not self.n_components:
            self.n_components = np.min([X.shape[0], X.shape[1]])

        m = np.mean(X, 0)
        X_c = X - m[np.newaxis, :]

        C = np.cov(X_c.T)

        evl, evc = np.linalg.eig(C)
        idx = evl.argsort()[::-1]
        evl, evc = evl[idx], evc[:, idx]
        evl, evc = evl[:self.n_components], evc[:, :self.n_components]

        self.evl, self.evc = evl, evc

    def transform(self, X):
        return np.matmul(X, self.evc)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
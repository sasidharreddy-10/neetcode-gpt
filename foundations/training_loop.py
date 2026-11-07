import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        w=np.zeros(X.shape[1])
        b=0
        for _ in range(epochs):
            x = np.dot(X, w) + b
            mse = np.mean((x-y)**2)

            # backward
            dl_dw = (2*(x-y)/len(X))@X
            dl_db = np.sum(2*(x-y)/len(X))

            w-=lr*dl_dw
            b-=lr*dl_db

        return (np.round(w, 5), round(float(b), 5))


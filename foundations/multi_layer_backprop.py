import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        z1 = np.dot(x, W1.T) + b1
        a1 = np.maximum(z1, 0)
        z2 = np.dot(a1, W2.T) + b2
        loss = np.mean((z2 - y_true)**2)

        # Backward
        dloss_dz2 = 2*(z2 - y_true)/len(y_true)
        dz2_da1 = W2.T
        da1_dz1 = np.where(z1, z1>0, 0)
        dz2_dw2 = a1
        dz1_dw1 = x
        dloss_dz1 = dz2_da1@dloss_dz2

        dloss_dw2 = np.outer(dloss_dz2, dz2_dw2)
        dloss_db2 = dloss_dz2

        dloss_dw1 = np.outer(dloss_dz1*da1_dz1, dz1_dw1)
        dloss_db1 = dloss_dz1*da1_dz1

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dloss_dw1, 4).tolist(),
            'db1': np.round(dloss_db1, 4).tolist(),
            'dW2': np.round(dloss_dw2, 4).tolist(),
            'db2': np.round(dloss_db2, 4).tolist(),
        }

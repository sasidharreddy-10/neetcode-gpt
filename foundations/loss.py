import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        clip_y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -(np.dot(y_true, np.log(clip_y_pred)) + np.dot((1-y_true), np.log(1-clip_y_pred)))/len(y_pred)
        return np.round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        clip_y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -(np.log(np.sum(clip_y_pred * y_true, axis=1)))/len(y_pred)
        print(loss)
        return np.round(np.sum(loss), 4)

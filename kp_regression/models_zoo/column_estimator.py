from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin


class ColumnEstimator(RegressorMixin, BaseEstimator):
    def __init__(self, column_idx: int = 0) -> None:
        self.column_idx = column_idx

    def fit(self, X: NDArray, y: NDArray):
        self.is_fitted_ = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        return X[:, self.column_idx]

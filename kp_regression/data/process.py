import typing as T

from numpy.typing import NDArray
from pandas import DataFrame
from kp_regression.data_pipe import KpData


class KpMixedLags(KpData):

    def process_data(
        self, df: DataFrame, params: dict
    ) -> T.Tuple[T.List[str], NDArray, NDArray]:
        pass

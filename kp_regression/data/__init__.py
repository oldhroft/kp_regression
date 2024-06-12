import typing as T

from kp_regression.data.process import KpMixedLags
from kp_regression.data_pipe import BaseData

DATA_FACTORY: T.Dict[str, T.Type[BaseData]] = {
    "KpMixedLags": KpMixedLags
}

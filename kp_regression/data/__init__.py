import typing as T

from numpy.typing import NDArray

from kp_regression.data.process import KpMixedLags, KpMixedLagsSeq
from kp_regression.data.postprocess import attach_kp_index_to_grid, clip_kp
from kp_regression.data_pipe import BaseData

DATA_FACTORY: T.Dict[str, T.Type[BaseData]] = {
    "KpMixedLags": KpMixedLags,
    "KpMixedLagsSeq": KpMixedLagsSeq
}

POST_PROCESS_FACTORY: T.Dict[str, T.Callable[[NDArray], NDArray]] = {
    "KpRound": attach_kp_index_to_grid,
    "KpClip": clip_kp,
    "Default": lambda x: x
}
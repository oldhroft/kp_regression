import typing as T

from numpy.typing import NDArray

from kp_regression.data.kp_5m_agg_mixed_lags import Kp5mAggMixedLags
from kp_regression.data.kp_5m_agg_mixed_lags_and_images import (
    Kp5mAggMixedLagsAndImages,
)
from kp_regression.data.kp_5m_mixed_lags_seq import Kp5mMixedLagsSeq
from kp_regression.data.kp_mixed_lags import KpMixedLags
from kp_regression.data.kp_mixed_lags_seq import KpMixedLagsSeq
from kp_regression.data.postprocess import attach_kp_index_to_grid, clip_kp
from kp_regression.data_pipe import BaseData

DATA_FACTORY: dict[str, type[BaseData]] = {
    "KpMixedLags": KpMixedLags,
    "KpMixedLagsSeq": KpMixedLagsSeq,
    "Kp5mAggMixedLags": Kp5mAggMixedLags,
    "Kp5mMixedLagsSeq": Kp5mMixedLagsSeq,
    "Kp5mAggMixedLagsAndImages": Kp5mAggMixedLagsAndImages,
}

POST_PROCESS_FACTORY: dict[str, T.Callable[[NDArray], NDArray]] = {
    "KpRound": attach_kp_index_to_grid,
    "KpClip": clip_kp,
    "Default": lambda x: x,
}

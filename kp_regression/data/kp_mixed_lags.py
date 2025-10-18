import typing as T
from pandas import DataFrame #type: ignore
from kp_regression.data_pipe import KpData, Dataset
from kp_regression.data_utils import add_diffs, add_lags
from .utils import process_data_standard

class KpMixedLags(KpData):
    def process_data(
        self,
        df: DataFrame,
        is_train: bool,
        lags_kp: int = 0,
        lags_h: int = 0,
        features_h: list = ["Dst"],
        features_other: list = [],
        n_targets: int = 8,
        hour_type: T.Optional[str] = None,
        diff_kp: bool = False,
        diff_features: T.List[str] = [],
        **kwargs,
    ) -> Dataset:
        return process_data_standard(
            data=df.sort_values(by="dttm").reset_index(drop=True),
            lags_kp=lags_kp,
            lags_h=lags_h,
            features_h=features_h,
            features_other=features_other,
            n_targets=n_targets,
            hour_type=hour_type,
            diff_kp=diff_kp,
            diff_features=diff_features,
        )

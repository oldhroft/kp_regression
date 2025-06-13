import typing as T

from pandas import DataFrame  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from kp_regression.data_pipe import Dataset, KpData

from kp_regression.data.utils import process_data_sequence


class KpMixedLagsSeq(KpData):
    def process_data(
        self,
        df: DataFrame,
        is_train: bool,
        lags_kp: int = 0,
        lags_h: int = 0,
        features_h: T.List[str] = ["Dst"],
        features_other: T.List[str] = [],
        n_targets: int = 8,
        scale: bool = False,
        **kwargs,
    ) -> Dataset:
        if is_train:
            scalers = StandardScaler(), StandardScaler(), StandardScaler()
            self.scalers = scalers
        else:
            if hasattr(self, "scalers"):
                scalers = self.scalers
            else:
                raise ValueError(
                    "Not fitted scalers yet, try first with is_train = True"
                )
        res, scalers = process_data_sequence(
            data=df.sort_values(by="dttm").reset_index(drop=True),
            is_train=is_train,
            lags_kp=lags_kp,
            lags_h=lags_h,
            features_h=features_h,
            features_other=features_other,
            n_targets=n_targets,
            scale=scale,
            scalers=scalers,
        )
        if is_train:
            self.scalers = scalers
        return res

import typing as T

from pandas import DataFrame  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from kp_regression.data.utils import process_data_sequence_5min
from kp_regression.data_pipe import Dataset, KpData5m


class Kp5mMixedLagsSeq(KpData5m):
    def process_data(
        self,
        df: DataFrame,
        df_1h: DataFrame,
        df_5m: DataFrame,
        is_train: bool,
        lags_5m: int = 0,
        lags_kp: int = 0,
        lags_h: int = 0,
        features_h: list = ["Dst"],
        features_5m: list = [],
        features_1h_ace: list = [],
        features_other: list = [],
        n_targets: int = 8,
        scale: bool = True,
        **kwargs,
    ) -> Dataset:
        if is_train:
            scalers = (
                StandardScaler(),
                StandardScaler(),
                StandardScaler(),
                StandardScaler(),
            )
            self.scalers = scalers
        else:
            if hasattr(self, "scalers"):
                scalers = self.scalers
            else:
                raise ValueError(
                    "Not fitted scalers yet, try first with is_train = True"
                )
        df = df.merge(df_1h[features_1h_ace + ["dttm"]], how="left", on="dttm")
        res, scalers = process_data_sequence_5min(
            data=df.sort_values(by="dttm").reset_index(drop=True),
            data_5m=df_5m.sort_values(by="dttm").reset_index(drop=True),
            is_train=is_train,
            lags_kp=lags_kp,
            lags_h=lags_h,
            lags_5m=lags_5m,
            features_h=features_h + features_1h_ace,
            features_5m=features_5m,
            features_other=features_other,
            n_targets=n_targets,
            scale=scale,
            scalers=scalers,
        )
        if is_train:
            self.scalers = scalers
        return res

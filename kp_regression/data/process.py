import typing as T

from numpy.typing import NDArray
from pandas import DataFrame, concat
from kp_regression.data_pipe import KpData, Dataset
from kp_regression.data_utils import add_lags


def process_data_standard(
    data: DataFrame,
    lags_kp: int,
    lags_h: int,
    features_h: list,
    features_other: list,
    n_targets: int,
):

    data = data.copy()
    data["t0_flg"] = (data["hour to"]) % 3 == 0
    data["t1_flg"] = (data["hour to"] + 1) % 3 == 0
    data["t2_flg"] = (data["hour to"] + 2) % 3 == 0

    data["hour_type"] = (
        data.t0_flg.map({True: "T0"})
        .fillna(data.t1_flg.map({True: "T1"}))
        .fillna(data.t2_flg.map({True: "T2"}))
    )

    meta_cols = ["dttm", "hour from", "hour to", "hour_type"]

    data["Kp"] = data["Kp*10"]

    flgs = ["t0_flg", "t1_flg", "t2_flg"]

    data_lagged, features_h_list = add_lags(
        data[features_h + features_other + flgs + meta_cols],
        subset=features_h,
        forward=False,
        lags=lags_h,
        trim=True,
    )

    data_lagged_3h_t0, features_3h_list = add_lags(
        data.loc[data.t0_flg, ["dttm", "Kp"]], subset=["Kp"], lags=lags_kp, trim=True
    )

    data_lagged_3h_t1, _ = add_lags(
        data.loc[data.t1_flg, ["dttm", "Kp"]]
        .assign(Kp=lambda x: x.Kp.shift())
        .iloc[1:],
        subset=["Kp"],
        lags=lags_kp,
        trim=True,
    )

    data_lagged_3h_t2, _ = add_lags(
        data.loc[
            data.t2_flg,
            [
                "dttm",
                "Kp",
            ],
        ]
        .assign(Kp=lambda x: x.Kp.shift())
        .iloc[1:],
        subset=["Kp"],
        lags=lags_kp,
        trim=True,
    )

    data_lagged_3h = (
        concat(
            [data_lagged_3h_t0, data_lagged_3h_t1, data_lagged_3h_t2],
            axis=0,
            ignore_index=True,
        )
        .sort_values(by="dttm")
        .reset_index(drop=True)
    )

    data_target_3h_t0, target_3h = add_lags(
        data.loc[data.t0_flg, ["dttm", "Kp"]],
        subset=["Kp"],
        lags=n_targets,
        trim=True,
        forward=True,
    )

    data_target_3h_t1, _ = add_lags(
        data.loc[data.t1_flg, ["dttm", "Kp"]]
        .assign(Kp=lambda x: x.Kp.shift(1))
        .iloc[1:],
        subset=["Kp"],
        lags=n_targets,
        trim=True,
        forward=True,
    )

    data_target_3h_t2, _ = add_lags(
        data.loc[
            data.t2_flg,
            [
                "dttm",
                "Kp",
            ],
        ]
        .assign(Kp=lambda x: x.Kp.shift())
        .iloc[1:],
        subset=["Kp"],
        lags=n_targets,
        trim=True,
        forward=True,
    )

    data_target_3h = (
        concat(
            [data_target_3h_t0, data_target_3h_t1, data_target_3h_t2],
            axis=0,
            ignore_index=True,
        )
        .sort_values(by="dttm")
        .loc[:, ["dttm"] + target_3h]
        .reset_index(drop=True)
    )

    result = data_lagged.merge(data_lagged_3h, how="inner", on="dttm").merge(
        data_target_3h, how="inner", on="dttm"
    )

    result_features = (
        features_other + features_h + features_h_list + ["Kp"] + features_3h_list + flgs
    )

    return Dataset(
        X=result[result_features].ffill().astype("float64").values,
        y=result[target_3h].ffill().astype("float64").values,
        meta=result[meta_cols],
        feature_names=result_features,
        target_names=target_3h,
    )


class KpMixedLags(KpData):

    def process_data(
        self,
        df: DataFrame,
        lags_kp: int,
        lags_h: int,
        features_h: list,
        features_other: list,
        n_targets: int,
    ) -> Dataset:
        return process_data_standard(
            data=df,
            lags_kp=lags_kp,
            lags_h=lags_h,
            features_h=features_h,
            features_other=features_other,
            n_targets=n_targets,
        )

import typing as T

from sklearn.preprocessing import StandardScaler

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
    hour_type: T.Optional[str] = None,
) -> Dataset:

    if hour_type is not None and hour_type not in ("T0", "T1", "T2"):

        raise ValueError(f"Unknown hour type {hour_type}")

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

    if hour_type is not None:
        result = result.loc[result.hour_type == hour_type]

    result_features = (
        ["Kp"] + features_other + features_h + features_h_list + features_3h_list + flgs
    )

    return Dataset(
        X=result[result_features].ffill().astype("float64").values,
        y=result[target_3h].ffill().astype("float64").values,
        meta=result[meta_cols],
        feature_names=result_features,
        target_names=target_3h,
        shape=(len(result_features),)
    )


class KpMixedLags(KpData):

    def process_data(
        self,
        df: DataFrame,
        is_train: bool,
        lags_kp: int,
        lags_h: int,
        features_h: list,
        features_other: list,
        n_targets: int,
        hour_type: T.Optional[str] = None,
    ) -> Dataset:
        return process_data_standard(
            data=df,
            lags_kp=lags_kp,
            lags_h=lags_h,
            features_h=features_h,
            features_other=features_other,
            n_targets=n_targets,
            hour_type=hour_type,
        )


def process_data_sequence(
    data: DataFrame,
    is_train: bool,
    lags_kp: int,
    lags_h: int,
    features_h: list,
    features_other: list,
    n_targets: int,
    scalers: T.Tuple[StandardScaler, StandardScaler, StandardScaler],
    scale: bool,
) -> T.Tuple[Dataset, T.Tuple[StandardScaler, StandardScaler, StandardScaler]]:

    scaler1, scaler2, scaler3 = scalers

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

    data_lagged_np = result[features_h + features_h_list].values
    data_lagged_3h_np = result[["Kp"] + features_3h_list].values

    data_flg = result[flgs].values.astype("float64")

    if scale:
        if is_train:
            data_lagged_np = scaler1.fit_transform(data_lagged_np)
            data_lagged_3h_np = scaler2.fit_transform(data_lagged_3h_np)
            data_flg = scaler2.fit_transform(data_flg)
        else:
            data_lagged_np = scaler1.transform(data_lagged_np)
            data_lagged_3h_np = scaler2.transform(data_lagged_3h_np)
            data_flg = scaler2.transform(data_flg)

    data_lagged_seq = data_lagged_np.reshape(-1, lags_h + 1, len(features_h)).transpose(
        0, 2, 1
    )

    data_lagged_3h_seq = data_lagged_3h_np.reshape(-1, lags_kp + 1, 1).transpose(
        0, 2, 1
    )

    result_features = [features_h, ["Kp"]]

    return Dataset(
        X=(data_lagged_seq, data_lagged_3h_seq, data_flg),
        y=result[target_3h].ffill().astype("float64").values,
        meta=result[meta_cols],
        feature_names=result_features,
        target_names=target_3h,
        shape=(
            (len(features_h), lags_h + 1),
            (1, lags_kp + 1),
            (3,)
        )
    ), (scaler1, scaler2, scaler3)


class KpMixedLagsSeq(KpData):

    def process_data(
        self,
        df: DataFrame,
        is_train: bool,
        lags_kp: int,
        lags_h: int,
        features_h: list,
        features_other: list,
        n_targets: int,
        scale: bool,
    ) -> Dataset:

        if is_train:
            scalers = StandardScaler(), StandardScaler(), StandardScaler()
        else:
            scalers = self.scalers

        res, scalers = process_data_sequence(
            data=df,
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

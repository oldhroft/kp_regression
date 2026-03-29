from pandas import DataFrame, concat  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from kp_regression.data_pipe import Dataset
from kp_regression.data_utils import add_diffs, add_lags


def process_data_standard(
    data: DataFrame,
    lags_kp: int,
    lags_h: int,
    features_h: list[str],
    features_other: list[str],
    n_targets: int,
    diff_features: list[str],
    diff_kp: bool,
    hour_type: str | None = None,
    target: str = "Kp",
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
    base_df = data[features_h + features_other + flgs + meta_cols].ffill()
    diff_features_list: list[str] = []
    if len(diff_features) > 0:
        base_df, diff_features_list = add_diffs(
            base_df, subset=diff_features, lags=1, trim=True, suffix_name="diff"
        )
    data_lagged, features_h_list = add_lags(
        base_df,
        subset=features_h + diff_features_list,
        forward=False,
        lags=lags_h,
        trim=True,
    )
    kp_list = ["Kp"]
    kp_diff_features: list[str] = []
    data_lagged_3h_t0 = data.loc[data.t0_flg, ["dttm", "Kp"]]
    if diff_kp:
        data_lagged_3h_t0, kp_diff_features = add_diffs(
            data_lagged_3h_t0, subset=["Kp"], lags=1, trim=True, suffix_name="diff"
        )
        kp_list = kp_list + kp_diff_features
    data_lagged_3h_t0, features_3h_list = add_lags(
        data_lagged_3h_t0, subset=kp_list, lags=lags_kp, trim=True
    )
    data_lagged_3h_t1 = (
        data.loc[data.t1_flg, ["dttm", "Kp"]].assign(Kp=lambda x: x.Kp.shift()).iloc[1:]
    )
    if diff_kp:
        data_lagged_3h_t1, _ = add_diffs(
            data_lagged_3h_t1, subset=["Kp"], lags=1, trim=True, suffix_name="diff"
        )
    data_lagged_3h_t1, _ = add_lags(
        data_lagged_3h_t1, subset=kp_list, lags=lags_kp, trim=True
    )
    data_lagged_3h_t2 = (
        data.loc[data.t2_flg, ["dttm", "Kp"]].assign(Kp=lambda x: x.Kp.shift()).iloc[1:]
    )
    if diff_kp:
        data_lagged_3h_t2, _ = add_diffs(
            data_lagged_3h_t2, subset=["Kp"], lags=1, trim=True, suffix_name="diff"
        )
    data_lagged_3h_t2, _ = add_lags(
        data_lagged_3h_t2, subset=kp_list, lags=lags_kp, trim=True
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

    if target == "Kp":
        data_target_3h_t0, target_cols = add_lags(
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
            data.loc[data.t2_flg, ["dttm", "Kp"]]
            .assign(Kp=lambda x: x.Kp.shift())
            .iloc[1:],
            subset=["Kp"],
            lags=n_targets,
            trim=True,
            forward=True,
        )
        data_target = (
            concat(
                [data_target_3h_t0, data_target_3h_t1, data_target_3h_t2],
                axis=0,
                ignore_index=True,
            )
            .sort_values(by="dttm")
            .reset_index(drop=True)
        )
    else:
        data_target, target_cols = add_lags(
            data.loc[:, ["dttm", target]],
            lags=n_targets,
            trim=True,
            forward=True,
            subset=[target],
        )

    result = data_lagged.merge(data_lagged_3h, how="inner", on="dttm").merge(
        data_target.loc[:, ["dttm"] + target_cols], how="inner", on="dttm"
    )
    if hour_type is not None:
        result = result.loc[result.hour_type == hour_type]
    result_features = (
        ["Kp"]
        + features_other
        + features_h
        + diff_features_list
        + features_h_list
        + features_3h_list
        + flgs
    )
    return Dataset(
        X=result[result_features].ffill().astype("float64").values,
        y=result[target_cols].ffill().astype("float64").values,
        meta=result[meta_cols],
        feature_names=result_features,
        target_names=target_cols,
        shape=(len(result_features),),
    )


def process_data_sequence(
    data: DataFrame,
    is_train: bool,
    lags_kp: int,
    lags_h: int,
    features_h: list,
    features_other: list,
    n_targets: int,
    scalers: tuple[StandardScaler, StandardScaler, StandardScaler],
    scale: bool,
    target: str = "Kp",
) -> tuple[Dataset, tuple[StandardScaler, StandardScaler, StandardScaler]]:
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
        data[features_h + features_other + flgs + meta_cols].ffill(),
        subset=features_h,
        forward=False,
        lags=lags_h,
        trim=True,
        sort_lags=True,
    )
    data_lagged_3h_t0, features_3h_list = add_lags(
        data.loc[data.t0_flg, ["dttm", "Kp"]],
        subset=["Kp"],
        lags=lags_kp,
        trim=True,
        sort_lags=True,
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

    if target == "Kp":
        data_target_3h_t0, target_cols = add_lags(
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
        data_target = (
            concat(
                [data_target_3h_t0, data_target_3h_t1, data_target_3h_t2],
                axis=0,
                ignore_index=True,
            )
            .sort_values(by="dttm")
            .reset_index(drop=True)
        )
    else:
        data_target, target_cols = add_lags(
            data.loc[:, ["dttm", target]],
            lags=n_targets,
            trim=True,
            forward=True,
            subset=[target],
        )

    result = data_lagged.merge(data_lagged_3h, how="inner", on="dttm").merge(
        data_target.loc[:, ["dttm"] + target_cols], how="inner", on="dttm"
    )
    data_lagged_np = result[features_h_list + features_h].ffill().values
    data_lagged_3h_np = result[features_3h_list + ["Kp"]].ffill().values
    data_flg = result[flgs].values.astype("float64")
    if scale:
        if is_train:
            data_lagged_np = scaler1.fit_transform(data_lagged_np)
            data_lagged_3h_np = scaler2.fit_transform(data_lagged_3h_np)
            data_flg = scaler3.fit_transform(data_flg)
        else:
            data_lagged_np = scaler1.transform(data_lagged_np)
            data_lagged_3h_np = scaler2.transform(data_lagged_3h_np)
            data_flg = scaler3.transform(data_flg)
    data_lagged_seq = data_lagged_np.reshape(-1, lags_h + 1, len(features_h)).transpose(
        0, 2, 1
    )
    data_lagged_3h_seq = data_lagged_3h_np.reshape(-1, lags_kp + 1, 1).transpose(
        0, 2, 1
    )
    result_features = [features_h, ["Kp"]]
    return Dataset(
        X=(data_lagged_seq, data_lagged_3h_seq, data_flg),
        y=result[target_cols].ffill().astype("float64").values,
        meta=result[meta_cols],
        feature_names=result_features,
        target_names=target_cols,
        shape=((len(features_h), lags_h + 1), (1, lags_kp + 1), (3,)),
    ), (scaler1, scaler2, scaler3)


def process_data_sequence_5min(
    data: DataFrame,
    data_5m: DataFrame,
    is_train: bool,
    lags_kp: int,
    lags_h: int,
    lags_5m: int,
    features_h: list,
    features_other: list,
    features_5m: list,
    n_targets: int,
    scalers: tuple[StandardScaler, StandardScaler, StandardScaler, StandardScaler],
    scale: bool,
    target: str = "Kp",
) -> tuple[
    Dataset, tuple[StandardScaler, StandardScaler, StandardScaler, StandardScaler]
]:
    scaler1, scaler2, scaler3, scaler4 = scalers
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
    meta_cols_5m = ["dttm"]
    data["Kp"] = data["Kp*10"]
    flgs = ["t0_flg", "t1_flg", "t2_flg"]
    data_lagged, features_h_list = add_lags(
        data[features_h + features_other + flgs + meta_cols].ffill(),
        subset=features_h,
        forward=False,
        lags=lags_h,
        trim=True,
        sort_lags=True,
    )
    data_lagged_5m, features_5m_list = add_lags(
        data_5m[features_5m + meta_cols_5m].ffill(),
        subset=features_5m,
        forward=False,
        lags=lags_5m,
        trim=True,
        sort_lags=True,
    )
    data_lagged_3h_t0, features_3h_list = add_lags(
        data.loc[data.t0_flg, ["dttm", "Kp"]],
        subset=["Kp"],
        lags=lags_kp,
        trim=True,
        sort_lags=True,
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

    if target == "Kp":
        data_target_3h_t0, target_cols = add_lags(
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
        data_target = (
            concat(
                [data_target_3h_t0, data_target_3h_t1, data_target_3h_t2],
                axis=0,
                ignore_index=True,
            )
            .sort_values(by="dttm")
            .reset_index(drop=True)
        )
    else:
        data_target, target_cols = add_lags(
            data.loc[:, ["dttm", target]],
            lags=n_targets,
            trim=True,
            forward=True,
            subset=[target],
        )

    intersecting_dttms = (
        data_lagged[["dttm"]]
        .merge(data_lagged_3h[["dttm"]], how="inner", on="dttm")
        .merge(data_target[["dttm"]], how="inner", on="dttm")
    )
    data_lagged_np = (
        intersecting_dttms.merge(data_lagged, how="left", on="dttm")[
            features_h_list + features_h
        ]
        .ffill()
        .values
    )
    data_lagged_3h_np = (
        intersecting_dttms.merge(data_lagged_3h, how="left", on="dttm")[
            features_3h_list + ["Kp"]
        ]
        .ffill()
        .values
    )
    data_flg = (
        intersecting_dttms.merge(data_lagged, how="left", on="dttm")[flgs]
        .ffill()
        .values.astype("float64")
    )
    data_lagged_5m_np = (
        intersecting_dttms.merge(data_lagged_5m, how="left", on="dttm")[
            features_5m_list + features_5m
        ]
        .ffill()
        .values
    )

    data_target_3h_np = (
        intersecting_dttms.merge(
            data_target.loc[:, ["dttm"] + target_cols], how="left", on="dttm"
        )[target_cols]
        .ffill()
        .values.astype("float64")
    )
    meta = intersecting_dttms.merge(data_lagged, how="left", on="dttm")[meta_cols]
    if scale:
        if is_train:
            data_lagged_np = scaler1.fit_transform(data_lagged_np)
            data_lagged_3h_np = scaler2.fit_transform(data_lagged_3h_np)
            data_lagged_5m_np = scaler3.fit_transform(data_lagged_5m_np)
            data_flg = scaler4.fit_transform(data_flg)
        else:
            data_lagged_np = scaler1.transform(data_lagged_np)
            data_lagged_3h_np = scaler2.transform(data_lagged_3h_np)
            data_lagged_5m_np = scaler3.transform(data_lagged_5m_np)
            data_flg = scaler4.transform(data_flg)
    data_lagged_seq = data_lagged_np.reshape(-1, lags_h + 1, len(features_h)).transpose(
        0, 2, 1
    )
    data_lagged_3h_seq = data_lagged_3h_np.reshape(-1, lags_kp + 1, 1).transpose(
        0, 2, 1
    )
    data_lagged_5m_seq = data_lagged_5m_np.reshape(
        -1, lags_5m + 1, len(features_5m)
    ).transpose(0, 2, 1)
    result_features = [features_h, ["Kp"], features_5m, flgs]
    return Dataset(
        X=(data_lagged_seq, data_lagged_3h_seq, data_lagged_5m_seq, data_flg),
        y=data_target_3h_np,
        meta=meta,
        feature_names=result_features,
        target_names=target_cols,
        shape=(
            (len(features_h), lags_h + 1),
            (1, lags_kp + 1),
            (len(features_5m), lags_5m + 1),
            (3,),
        ),
    ), (scaler1, scaler2, scaler3, scaler4)

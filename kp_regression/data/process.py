import typing as T

from sklearn.preprocessing import StandardScaler # type: ignore

from pandas import DataFrame, concat # type: ignore
from kp_regression.data_pipe import KpData, Dataset, KpData5m
from kp_regression.data_utils import add_lags, add_diffs


def process_data_standard(
    data: DataFrame,
    lags_kp: int,
    lags_h: int,
    features_h: T.List[str],
    features_other: T.List[str],
    n_targets: int,
    diff_features: T.List[str],
    diff_kp: bool,
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

    base_df = data[features_h + features_other + flgs + meta_cols].ffill()

    diff_features_list: T.List[str] = []
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
    kp_diff_features: T.List[str] = []
    data_lagged_3h_t1 = data.loc[data.t1_flg, ["dttm", "Kp"]]
    if diff_kp:
        data_lagged_3h_t1, kp_diff_features = add_diffs(
            data_lagged_3h_t1, subset=["Kp"], lags=1, trim=True, suffix_name="diff"
        )
        kp_list = kp_list + kp_diff_features

    data_lagged_3h_t1, features_3h_list = add_lags(
        data_lagged_3h_t1, subset=kp_list, lags=lags_kp, trim=True
    )

    data_lagged_3h_t0 = data.loc[data.t0_flg, ["dttm", "Kp"]]

    if diff_kp:
        data_lagged_3h_t0, _ = add_diffs(
            data_lagged_3h_t0, subset=["Kp"], lags=1, trim=True, suffix_name="diff"
        )

    data_lagged_3h_t0, features_3h_list = add_lags(
        data_lagged_3h_t0, subset=kp_list, lags=lags_kp, trim=True
    )

    data_lagged_3h_t2 = data.loc[data.t2_flg, ["dttm", "Kp"]]
    if diff_kp:
        data_lagged_3h_t2, _ = add_diffs(
            data_lagged_3h_t2, subset=["Kp"], lags=1, trim=True, suffix_name="diff"
        )

    data_lagged_3h_t2, _ = add_lags(
        data_lagged_3h_t2.assign(Kp=lambda x: x.Kp.shift()).iloc[1:],
        subset=kp_list,
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
        data.loc[data.t2_flg, ["dttm", "Kp"]]
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

    # Use both original features and their diffs in the final dataset
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
        y=result[target_3h].ffill().astype("float64").values,
        meta=result[meta_cols],
        feature_names=result_features,
        target_names=target_3h,
        shape=(len(result_features),),
    )


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
        **kwargs
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
        y=result[target_3h].ffill().astype("float64").values,
        meta=result[meta_cols],
        feature_names=result_features,
        target_names=target_3h,
        shape=((len(features_h), lags_h + 1), (1, lags_kp + 1), (3,)),
    ), (scaler1, scaler2, scaler3)


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
        **kwargs
    ) -> Dataset:

        if is_train:
            scalers = StandardScaler(), StandardScaler(), StandardScaler()
            self.scalers = scalers
        else:
            if hasattr(self, "scalers"):
                scalers = self.scalers
            else:
                raise ValueError("Not fitted scalers yet, try first with is_train = True")

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


class Kp5mAggMixedLags(KpData5m):
    def process_data(
        self,
        df: DataFrame,
        df_1h: DataFrame,
        df_5m: DataFrame,
        is_train: bool,
        lags_kp: int = 0,
        lags_h: int = 0,
        features_h: T.List[str] = ["Dst"],
        features_1h_ace: T.List[str] = [],
        features_5m_agg: T.List[str] = [],
        agg_list: T.List[str] =[],
        agg_quantiles: T.List[float] = [],
        features_other: T.List[str] = [],
        n_targets: int = 8,
        diff_kp: bool = False,
        diff_features: T.List[str] = [],
        diff_features_5m: T.List[str] = [],
        **kwargs
    ) -> Dataset:

        from pandas import Grouper
        from numpy import NaN

        save_cols = ["dttm", "hour from", "hour to", "Kp*10"]

        df_5m = df_5m.copy()
        df_1h = df_1h.copy()
        df = df.copy()

        df_5m[features_5m_agg] = df_5m[features_5m_agg].where(
            df_5m[features_5m_agg] > -999.9, NaN
        )

        df_1h[features_1h_ace] = df_1h[features_1h_ace].where(
            df_5m[features_1h_ace] > -999.9, NaN
        )

        # Add diffs to df_5m if requested
        diff_features_5m_list: T.List[str] = []
        if len(diff_features_5m) > 0:
            df_5m, diff_features_5m_list = add_diffs(
                df_5m, subset=diff_features_5m, lags=1, trim=True, suffix_name="diff5m"
            )
            features_5m_agg = features_5m_agg + diff_features_5m_list

        df_5m_agg = concat(
            [
                (
                    df_5m.groupby(Grouper(key="dttm", freq="h"))[feat]
                    .agg(agg_list)
                    .join(
                        df_5m.groupby(Grouper(key="dttm", freq="h"))[feat]
                        .quantile(agg_quantiles)
                        .unstack(level=1)
                        .add_prefix("q")
                    )
                    .add_prefix(f"{feat}_")
                )
                for feat in features_5m_agg
            ],
            axis=1,
        )

        agg_features = list(df_5m_agg.columns)

        intersecting_columns = set(features_1h_ace).intersection(features_h)

        if len(intersecting_columns) > 0:
            raise ValueError("Identical column in two different sources")

        df_result = (
            df[save_cols + features_h]
            .merge(df_5m_agg[agg_features].reset_index(), how="left", on="dttm")
            .merge(df_1h[features_1h_ace + ["dttm"]], how="left", on="dttm")
        )

        return process_data_standard(
            data=df_result,
            lags_kp=lags_kp,
            lags_h=lags_h,
            features_h=features_h + agg_features + features_1h_ace,
            features_other=features_other,
            n_targets=n_targets,
            hour_type=None,
            diff_kp=diff_kp,
            diff_features=diff_features,
        )


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
    scalers: T.Tuple[StandardScaler, StandardScaler, StandardScaler, StandardScaler],
    scale: bool,
) -> T.Tuple[
    Dataset, T.Tuple[StandardScaler, StandardScaler, StandardScaler, StandardScaler]
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

    intersecting_dttms = (
        data_lagged[["dttm"]]
        .merge(data_lagged_3h[["dttm"]], how="inner", on="dttm")
        .merge(data_target_3h[["dttm"]], how="inner", on="dttm")
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
        intersecting_dttms.merge(data_target_3h, how="left", on="dttm")[target_3h]
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
        target_names=target_3h,
        shape=(
            (len(features_h), lags_h + 1),
            (1, lags_kp + 1),
            (len(features_5m), lags_5m + 1),
            (3,),
        ),
    ), (scaler1, scaler2, scaler3, scaler4)


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
        **kwargs
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
                raise ValueError("Not fitted scalers yet, try first with is_train = True")


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

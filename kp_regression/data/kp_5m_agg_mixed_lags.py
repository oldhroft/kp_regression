
from pandas import DataFrame  # type: ignore

from kp_regression.data.utils import process_data_standard
from kp_regression.data_pipe import Dataset, KpData5m
from kp_regression.data_utils import add_diffs


class Kp5mAggMixedLags(KpData5m):
    def process_data(
        self,
        df: DataFrame,
        df_1h: DataFrame,
        df_5m: DataFrame,
        is_train: bool,
        lags_kp: int = 0,
        lags_h: int = 0,
        features_h: list[str] = ["Dst"],
        features_1h_ace: list[str] = [],
        features_5m_agg: list[str] = [],
        agg_list: list[str] = [],
        agg_quantiles: list[float] = [],
        features_other: list[str] = [],
        n_targets: int = 8,
        diff_kp: bool = False,
        diff_features: list[str] = [],
        diff_features_5m: list[str] = [],
        **kwargs,
    ) -> Dataset:
        from numpy import nan
        from pandas import Grouper, concat

        save_cols = ["dttm", "hour from", "hour to", "Kp*10"]
        df_5m = df_5m.copy()
        df_1h = df_1h.copy()
        df = df.copy()
        df_5m[features_5m_agg] = df_5m[features_5m_agg].where(
            df_5m[features_5m_agg] > -999.9, nan
        )
        df_1h[features_1h_ace] = df_1h[features_1h_ace].where(
            df_5m[features_1h_ace] > -999.9, nan
        )
        diff_features_5m_list: list[str] = []
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

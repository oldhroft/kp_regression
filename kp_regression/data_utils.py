import typing as T
from pandas import DataFrame, Index


def _trim(df: DataFrame, forward: bool, trim: bool, lags: int) -> DataFrame:
    if trim and forward:
        return df.iloc[:-lags]
    elif trim:
        return df.iloc[lags:]
    else:
        return df


def add_lags(
    df: DataFrame,
    subset: T.Optional[T.Union[str, T.List[str]]] = None,
    forward: bool = False,
    lags: int = 1,
    trim: bool = False,
    suffix_name: str = None,
) -> T.Tuple[DataFrame.T.List[str]]:
    if suffix_name is None:
        suffix_name = "lead" if forward else "lag"

    x = df.copy()

    digits = len(str(lags))

    columns = []

    if not isinstance(lags, int):
        raise ValueError(f"Lags should be int, {type(lags)} type prodided")
    elif lags < 0:
        raise ValueError(f"Lags should be non-negative")
    elif lags == 0:
        return x, []
    elif subset is None:
        for i in range(1, lags + 1):
            lag = -i if forward else i
            index = str(i).zfill(digits)
            column_suffix = f"_{suffix_name}_{index}"

            x = x.join(x.shift(lag).add_suffix(column_suffix))

        columns = x.columns.tolist()

    elif isinstance(subset, list):
        for i in range(1, lags + 1):
            lag = -i if forward else i
            index = str(i).zfill(digits)
            column_suffix = f"_{suffix_name}_{index}"
            tmp = x.loc[:, subset].shift(lag).add_suffix(column_suffix)
            columns.extend(tmp.columns)
            x = x.join(tmp)

    elif isinstance(subset, str):
        for i in range(1, lags + 1):
            lag = -i if forward else i
            index = str(i).zfill(digits)
            column_name = f"{subset}_{suffix_name}_{index}"
            columns.append(column_name)

            x = x.join(x.loc[:, subset].shift(lag).rename(column_name))
    else:
        raise ValueError(f"Subset should be str or list, provided type {type(subset)}")

    return _trim(x, forward, trim, lags), columns


def rolling_agg(
    data: DataFrame,
    windows: T.List[int],
    functions: T.List[str],
    subset: T.List[str],
    return_features: bool = True,
) -> DataFrame:
    data = data.copy()
    features = []
    index_subset = Index(subset)
    for window in windows:
        for function in functions:
            suffix = f"_rolling_{window}_{function}"
            features.extend(list(index_subset + suffix))
            if function.startswith("quantile"):
                qnt = int(function.split("_")[1]) / 100
                agg = (
                    data[subset].rolling(window, min_periods=0).quantile(qnt).fillna(0)
                )
            else:
                agg = (
                    data[subset].rolling(window, min_periods=0).agg(function).fillna(0)
                )

            data = data.join(agg.add_suffix(suffix))

    if return_features:
        return data, features
    else:
        return data

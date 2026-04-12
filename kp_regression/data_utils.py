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
    subset: str | list[str] | None = None,
    forward: bool = False,
    lags: int = 1,
    lags_from: int = 0,
    trim: bool = False,
    suffix_name: str | None = None,
    sort_lags: bool = False,
) -> tuple[DataFrame, list[str]]:
    if suffix_name is None:
        suffix_name = "lead" if forward else "lag"

    x = df.copy()

    digits = len(str(lags))

    columns: list[str] = []
    sort_order: dict[str, tuple[int, str]] = {}

    if subset is None:
        subset = list(df.columns)

    if not isinstance(lags, int):
        raise ValueError(f"Lags should be int, {type(lags)} type prodided")
    elif lags < 0:
        raise ValueError("Lags should be non-negative")
    elif lags == 0:
        return x, []
    elif isinstance(subset, list):
        for i in range(lags_from + 1, lags + 1):
            lag = -i if forward else i
            index = str(i).zfill(digits)
            column_suffix = f"_{suffix_name}_{index}"
            tmp = x.loc[:, subset].shift(lag).add_suffix(column_suffix)
            columns.extend(tmp.columns)

            zipped = list(map(lambda x: (i, x), subset))
            new_pairs = dict(zip(list(tmp.columns), zipped))
            sort_order = dict(sort_order, **new_pairs)

            x = x.join(tmp)

    elif isinstance(subset, str):
        for i in range(lags_from + 1, lags + 1):
            lag = -i if forward else i
            index = str(i).zfill(digits)
            column_name = f"{subset}_{suffix_name}_{index}"
            columns.append(column_name)
            sort_order[column_name] = (i, subset)
            x = x.join(x.loc[:, subset].shift(lag).rename(column_name))
    else:
        raise ValueError(f"Subset should be str or list, provided type {type(subset)}")

    if sort_lags:
        columns = sorted(columns, key=lambda x: sort_order[x], reverse=True)
    return _trim(x, forward, trim, lags), columns


def add_diffs(
    df: DataFrame,
    subset: str | list[str] | None = None,
    lags: int = 1,
    trim: bool = False,
    suffix_name: str | None = None,
) -> tuple[DataFrame, list[str]]:
    if suffix_name is None:
        suffix_name = "diff"

    x = df.copy()
    digits = len(str(lags))
    columns = []

    if not isinstance(lags, int):
        raise ValueError(f"Lags should be int, {type(lags)} type provided")
    elif lags < 1:
        raise ValueError("Lags should be positive")
    elif lags == 0:
        return x, []

    if subset is None:
        subset_cols = x.columns.tolist()
    elif isinstance(subset, str):
        subset_cols = [subset]
    elif isinstance(subset, list):
        subset_cols = subset
    else:
        raise ValueError(f"Subset should be str or list, provided type {type(subset)}")

    for i in range(1, lags + 1):
        index = str(i).zfill(digits)
        suffix = f"_{suffix_name}_{index}"
        diffed = x[subset_cols].diff(i).add_suffix(suffix)
        columns.extend(diffed.columns)
        x = x.join(diffed)

    if trim:
        x = x.iloc[lags:]

    return x, columns


def rolling_agg(
    data: DataFrame,
    windows: list[int],
    functions: list[str],
    subset: list[str],
    return_features: bool = True,
) -> DataFrame | tuple[DataFrame, list[str]]:
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

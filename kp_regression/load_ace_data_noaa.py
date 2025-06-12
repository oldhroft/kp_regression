import datetime
import logging
import os
import typing as T
from urllib.error import HTTPError

import click
import polars as pl
import polars.type_aliases as pdt
from joblib import Parallel, delayed  # type: ignore
from tqdm import tqdm  # type: ignore

from kp_regression.logging_utils import config_logger

logger = logging.getLogger()



INIT_SCHEMA: T.Dict[str, pdt.plt.SchemaDict] = {
    "ace_mag": {
        "year": pl.Int32,
        "month": pl.Int32,
        "day": pl.Int32,
        "time": pl.String,
        "julian_day": pl.Int32,
        "seconds_of_day": pl.Int64,
        "status": pl.Int16,
        "Bx": pl.Float64,
        "By": pl.Float64,
        "Bz": pl.Float64,
        "Bt": pl.Float64,
        "lat": pl.Float64,
        "lon": pl.Float64,
    },
    "ace_swepam": {
        "year": pl.Int32,
        "month": pl.Int32,
        "day": pl.Int32,
        "time": pl.String,
        "julian_day": pl.Int32,
        "seconds_of_day": pl.Int64,
        "status": pl.Int16,
        "H_den_SWP": pl.Float64,
        "SW_spd": pl.Float64,
        "Trr_SWP": pl.Float64,
    },
}

SCHEMA: T.Dict[str, pdt.plt.SchemaDict] = {
    "ace_mag": {
        "dttm": pl.Datetime,
        "status": pl.Int16,
        "Bx": pl.Float64,
        "By": pl.Float64,
        "Bz": pl.Float64,
        "Bt": pl.Float64,
        "lat": pl.Float64,
        "lon": pl.Float64,
    },
    "ace_swepam": {
        "dttm": pl.Datetime,
        "status": pl.Int16,
        "H_den_SWP": pl.Float64,
        "SW_spd": pl.Float64,
        "Trr_SWP": pl.Float64,
    },
}

FILE_FMT = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/{agg_level}/{dt}_{data_type}_{freq}.txt"

READ_CFG = dict(
    comment_prefix="#", skip_lines=2, has_header=False, new_columns=["data"]
)

TYPES = ["ace_swepam", "ace_mag"]
FREQS = ["1m", "1h"]

OUT_FMT = {"1h": "%Y%m", "1m": "%Y%m%d"}

DataOptions = T.Literal["ace_swepam", "ace_mag"]
FreqOptions = T.Literal["1m", "1h"]


def process_data(
    data: pl.LazyFrame,
    init_schema: pdt.plt.SchemaDict,
    schema: pdt.plt.SchemaDict,
    from_date: str,
    to_date: str,
) -> pl.LazyFrame:

    from_dttm = datetime.datetime.fromisoformat(from_date)
    to_dttm = datetime.datetime.fromisoformat(to_date)

    return (
        data.with_columns(
            pl.col("data")
            .str.strip_chars_end(" ")
            .str.replace_all("\s+", " ")
            .str.split(" ")
            .alias("data_list")
        )
        .select(pl.col("data_list").list.to_struct(fields=list(init_schema.keys())))
        .unnest("data_list")
        .cast(init_schema) # type: ignore
        .with_columns(
            pl.col("time").str.slice(0, 2).cast(pl.Int32).alias("hour"),
            pl.col("time").str.slice(2, 4).cast(pl.Int32).alias("minute"),
        )
        .with_columns(
            pl.datetime(
                pl.col("year").cast(pl.Int32),
                pl.col("month").cast(pl.Int32),
                pl.col("day").cast(pl.Int32),
                pl.col("hour"),
                pl.col("minute"),
            ).alias("dttm")
        )
        .select(list(schema.keys()))
        .cast(schema) # type: ignore
        .filter(pl.col("dttm").is_between(from_dttm, to_dttm))
    )


def load_data(path: str, **read_cfg) -> pl.LazyFrame:

    try:
        df = pl.read_csv(path, **read_cfg).lazy()
    except HTTPError as e:
        logger.info("not found skipping file %s", path)
        df = pl.LazyFrame([], schema=pl.Schema({"data": pl.String}))
    return df


def get_file_range(
    from_date: str,
    to_date: str,
    data_type: DataOptions = "ace_swepam",
    freq: FreqOptions = "1h",
) -> T.List[str]:

    agg_level = "monthly" if freq == "1h" else "daily"

    from_dttm = datetime.datetime.fromisoformat(from_date)
    to_dttm = datetime.datetime.fromisoformat(to_date)

    out_fmt = OUT_FMT[freq]

    if freq == "1m":
        days = (to_dttm - from_dttm).days

        return [
            FILE_FMT.format(
                dt=(from_dttm + datetime.timedelta(days=i)).strftime(out_fmt),
                data_type=data_type,
                agg_level=agg_level,
                freq=freq,
            )
            for i in range(days + 1)
        ]
    else:
        from dateutil.relativedelta import relativedelta

        from_dt = from_dttm.date().replace(day=1)
        to_dt = to_dttm.date().replace(day=1)

        months = (to_dt.year - from_dt.year) * 12 + to_dt.month - from_dt.month
        return [
            FILE_FMT.format(
                dt=(from_dttm + relativedelta(months=i)).strftime(out_fmt),
                data_type=data_type,
                agg_level=agg_level,
                freq=freq,
            )
            for i in range(months + 1)
        ]


def download_ace_data(
    from_date: str, to_date: str, output_folder: str, data_type: str, freq: str
) -> None:

    if data_type not in TYPES:
        raise ValueError(
            f"Unknown data: {data_type}, possible data types: {';'.join(TYPES)}"
        )
    if freq not in FREQS:
        raise ValueError(
            f"Unknown data: {freq}, possible data types: {';'.join(FREQS)}"
        )

    config_logger(logger, stdout=False)

    assert data_type in TYPES, "Data type should be instance of types"

    data_type = T.cast(DataOptions, data_type)

    rng = get_file_range(from_date=from_date, to_date=to_date, data_type=data_type)

    if len(rng) == 0:
        logger.info("No files for this period found, exiting")
        return

    for link in rng:
        logger.info("Link %s is added to the download list", link)

    logger.info(
        "Collecting data for %s (freq %s) from %s to %s, total %s files",
        data_type,
        freq,
        from_date,
        to_date,
        len(rng),
    )

    data_lst: T.List[pl.LazyFrame] = Parallel(n_jobs=-1)(
        map(
            lambda x: delayed(load_data)(x, **READ_CFG),
            tqdm(rng),
        )
    )

    logger.info("Concatenating data, got %s items", len(data_lst))

    data = pl.concat(data_lst)

    data = process_data(
        data,
        init_schema=INIT_SCHEMA[data_type],
        schema=SCHEMA[data_type],
        from_date=from_date,
        to_date=to_date,
    )

    path = os.path.join(
        output_folder, f"data_{data_type}_{freq}_{from_date}_{to_date}.parquet"
    )
    os.makedirs(output_folder, exist_ok=True)

    stat = data.select(
        pl.len().alias("cnt_records"),
        pl.min("dttm").alias("min_dttm"),
        pl.max("dttm").alias("max_dttm"),
    ).collect()

    logger.info("Total records in data %s", stat["cnt_records"].item())
    logger.info("Min dttm in data %s", stat["min_dttm"].item())
    logger.info("Max dttm in data %s", stat["max_dttm"].item())

    logger.info("Saving data to %s", path)

    data.sink_parquet(path)


@click.command()
@click.option(
    "--from_date", help="Start date to download data", type=click.STRING, required=True
)
@click.option(
    "--to_date", help="End date to download data", type=click.STRING, required=True
)
@click.option(
    "--output_folder", help="Path to output folder", type=click.STRING, required=True
)
@click.option(
    "--data_type",
    help=f"Data to download: {';'.join(TYPES)}",
    type=click.STRING,
    required=True,
)
@click.option(
    "--freq",
    help=f"Which frequency to download {';'.join(FREQS)}",
    type=click.STRING,
    required=True,
)
def download_ace_data_run(
    from_date: str, to_date: str, output_folder: str, data_type: str, freq: str
) -> None:
    download_ace_data(
        from_date=from_date,
        to_date=to_date,
        output_folder=output_folder,
        data_type=data_type,
        freq=freq,
    )

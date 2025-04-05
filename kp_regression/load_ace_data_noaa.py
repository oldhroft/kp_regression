import os
import polars as pl
import datetime
from urllib.error import HTTPError

from joblib import Parallel, delayed
from tqdm import tqdm

import click

import typing as T

from kp_regression.logging_utils import config_logger

import logging

logger = logging.getLogger()


INIT_SCHEMA = {
    "ace_mag_1m": pl.Schema(
        {
            "year": pl.Int32(),
            "month": pl.Int32(),
            "day": pl.Int32(),
            "time": pl.String(),
            "julian_day": pl.Int32(),
            "seconds_of_day": pl.Int64(),
            "status": pl.Int16(),
            "Bx": pl.Float64(),
            "By": pl.Float64(),
            "Bz": pl.Float64(),
            "Bt": pl.Float64(),
            "lat": pl.Float64(),
            "lon": pl.Float64(),
        }
    ),
    "ace_swepam_1m": pl.Schema(
        {
            "year": pl.Int32(),
            "month": pl.Int32(),
            "day": pl.Int32(),
            "time": pl.String(),
            "julian_day": pl.Int32(),
            "seconds_of_day": pl.Int64(),
            "status": pl.Int16(),
            "H_den_SWP": pl.Float64(),
            "SW_spd": pl.Float64(),
            "Trr_SWP": pl.Float64(),
        }
    ),
}

SCHEMA = {
    "ace_mag_1m": pl.Schema(
        {
            "dttm": pl.Datetime(),
            "status": pl.Int16(),
            "Bx": pl.Float64(),
            "By": pl.Float64(),
            "Bz": pl.Float64(),
            "Bt": pl.Float64(),
            "lat": pl.Float64(),
            "lon": pl.Float64(),
        }
    ),
    "ace_swepam_1m": pl.Schema(
        {
            "dttm": pl.Datetime(),
            "status": pl.Int16(),
            "H_den_SWP": pl.Float64(),
            "SW_spd": pl.Float64(),
            "Trr_SWP": pl.Float64(),
        }
    ),
}

FILE_FMT = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/{dt}_{data_type}.txt"

READ_CFG = dict(
    comment_prefix="#", skip_lines=2, has_header=False, new_columns=["data"]
)

TYPES = ["ace_swepam_1m", "ace_mag_1m"]

DataOptions = T.Literal["ace_swepam_1m", "ace_mag_1m"]


def process_data(
    data: pl.LazyFrame, init_schema: pl.Schema, schema: pl.Schema
) -> pl.LazyFrame:

    return (
        data.with_columns(
            pl.col("data")
            .str.strip_chars_end(" ")
            .str.replace_all("\s+", " ")
            .str.split(" ")
            .alias("data_list")
        )
        .select(pl.col("data_list").list.to_struct(fields=init_schema.names()))
        .unnest("data_list")
        .cast(init_schema)
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
        .select(schema.names())
        .cast(schema)
    )


def load_data(
    path: str, init_schema: pl.Schema, schema: pl.Schema, **read_cfg
) -> pl.LazyFrame:

    try:
        df_raw = pl.read_csv(path, **read_cfg).lazy()
    except HTTPError as e:
        logger.info("not found skipping file %s", path)
        df_raw = pl.LazyFrame([], schema=pl.Schema({"data": pl.String}))

    df = process_data(df_raw, init_schema=init_schema, schema=schema)

    return df


def get_file_range(
    from_date: str,
    to_date: str,
    out_fmt: str = "%Y%m%d",
    data_type: DataOptions = "ace_swepam_1m",
) -> T.List[str]:

    from_dttm = datetime.datetime.fromisoformat(from_date)
    to_dttm = datetime.datetime.fromisoformat(to_date)

    days = (to_dttm - from_dttm).days

    return [
        FILE_FMT.format(
            dt=(from_dttm + datetime.timedelta(days=i)).strftime(out_fmt),
            data_type=data_type,
        )
        for i in range(days)
    ]


def download_ace_data(
    from_date: str, to_date: str, output_folder: str, data_type: str
) -> None:

    if data_type not in TYPES:
        raise ValueError(
            f"Unknown data: {data_type}, possible data types: {';'.join(TYPES)}"
        )

    config_logger(logger, stdout=False)

    rng = get_file_range(from_date=from_date, to_date=to_date, data_type=data_type)

    logger.info("Collecting data for %s from %s to %s", data_type, from_date, to_date)

    data_lst: T.List[pl.LazyFrame] = Parallel(n_jobs=-1)(
        map(
            lambda x: delayed(load_data)(
                x,
                init_schema=INIT_SCHEMA[data_type],
                schema=SCHEMA[data_type],
                **READ_CFG,
            ),
            tqdm(rng),
        )
    )

    logger.info("Concatenating data, got %s items", len(data_lst))
    data = pl.concat(data_lst)

    path = os.path.join(
        output_folder, f"data_{data_type}_{from_date}_{to_date}.parquet"
    )
    os.makedirs(output_folder, exist_ok=True)

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
def download_ace_data_run(
    from_date: str, to_date: str, output_folder: str, data_type: str
) -> None:
    download_ace_data(
        from_date=from_date,
        to_date=to_date,
        output_folder=output_folder,
        data_type=data_type,
    )

import polars as pl
import requests

import click

from urllib.parse import urljoin
from bs4 import BeautifulSoup

from joblib import Parallel, delayed
from tqdm import tqdm

from pyhdf.HDF import HDF, HC
from pyhdf.VS import VS, VD

import re
import os

from numpy.typing import NDArray

import typing as T


from kp_regression.logging_utils import config_logger

import logging

logger = logging.getLogger()


BROWSE_URL = "https://izw1.caltech.edu/ACE/ASC/DATA/browse-data/"
DATA_TYPES = ["ace_br_5min_avg", "ace_br_1hr_avg", "ace_br_1dy_avg"]
DATE_FMT = "%a %b %d %H:%M:%S %Y"

TypeLiteral = T.Literal["ace_br_5min_avg", "ace_br_1hr_avg", "ace_br_1dy_avg"]


def decode_bytes_dttm(arr: T.Union[NDArray, list]) -> str:

    bytes_string = bytes(arr)

    decoded_string = (
        bytes_string.decode("utf-8", errors="ignore")
        .strip()
        .replace("\x00", "")
        .strip("\n")
    )

    return decoded_string


def vdata_to_pl(vdata: VD) -> pl.LazyFrame:
    field_names = vdata._fields
    return pl.DataFrame(vdata[:], schema=field_names, orient="row").lazy()


def extract_data_from_hdf(path: str, data_type: TypeLiteral) -> pl.DataFrame:
    file = HDF(path, HC.READ)
    v = file.vstart()

    vdata = v.attach(data_type)

    result = vdata_to_pl(vdata)

    vdata.detach()
    v.end()
    return result


def load_file(url: str, dirname: str, use_cache: bool) -> str:

    fname = url.split("/")[-1]

    fpath = os.path.join(dirname, fname)

    if use_cache and os.path.exists(fpath):
        logger.info("Skipping %s, because got cache!", fpath)
        return fpath

    logger.info("Loading %s", fpath)

    response = requests.get(url)
    response.raise_for_status()
    with open(fpath, "wb") as f:
        f.write(response.content)

    return fpath


def process_hdf_file_from_url(
    url: str, dirname: str, data_type: TypeLiteral, use_cache: bool
) -> pl.DataFrame:

    fpath = load_file(url=url, dirname=dirname, use_cache=use_cache)
    df = extract_data_from_hdf(fpath, data_type=data_type)

    return df


def process_df(df: pl.LazyFrame) -> pl.LazyFrame:

    return df.with_columns(
        pl.col("timestr")
        .map_elements(decode_bytes_dttm, return_dtype=pl.String())
        .str.to_datetime(DATE_FMT)
        .alias("dttm")
    )


def build_file_list(from_year: int, to_year: int) -> T.List[str]:

    NAME_PATTERN = re.compile(r"ACE\_BROWSE\_(\d+)")

    def get_year(fpath: str) -> int:

        res = re.match(NAME_PATTERN, fpath.split("/")[-1])

        if res is None:
            raise ValueError("File name does not contain year")

        return int(res.group(1))

    def get_hrefs(url: str) -> T.List[str]:

        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all 'a' tags with 'href' attributes
        links = soup.find_all("a", href=True)

        # Extract directory names (hrefs ending with '/')
        return [link["href"] for link in links if link["href"]]

    logger.info("Building initial urls list")
    all_urls = get_hrefs(BROWSE_URL)

    browse_urls = list(
        map(
            lambda x: urljoin(BROWSE_URL, x),
            filter(lambda x: x.startswith("browse"), all_urls),
        )
    )
    logger.info("Building file urls list")

    # Output the list of directories
    sub_dirs = Parallel(n_jobs=40, backend="threading")(
        map(delayed(get_hrefs), tqdm(browse_urls))
    )

    files = list(
        map(
            lambda x: urljoin(x[0], list(filter(lambda y: y.endswith("HDF"), x[1]))[0]),
            zip(browse_urls, sub_dirs),
        )
    )
    logger.info("Filtering by year")

    return list(filter(lambda x: from_year <= get_year(x) <= to_year, files))


def download_ace_data(
    from_year: int,
    to_year: int,
    output_folder: str,
    output_folder_raw: str,
    data_type: str,
    use_cache: bool,
) -> None:

    if data_type not in DATA_TYPES:
        raise ValueError(
            f"Unknown data: {data_type}, possible data types: {';'.join(DATA_TYPES)}"
        )

    config_logger(logger, stdout=False)

    file_list = build_file_list(from_year=from_year, to_year=to_year)

    logger.info("Got %s files to load", len(file_list))

    logger.info("Start data loading...")

    os.makedirs(output_folder_raw, exist_ok=True)
    df_list: T.List[pl.LazyFrame] = Parallel(n_jobs=30, backend="threading")(
        map(
            delayed(
                lambda x: process_hdf_file_from_url(
                    x,
                    dirname=output_folder_raw,
                    data_type=data_type,
                    use_cache=use_cache,
                )
            ),
            tqdm(file_list),
        )
    )

    logger.info("Collecting and processing data")

    df = pl.concat(df_list).pipe(process_df)

    fname = os.path.join(
        output_folder, f"ace_data_{data_type}_{from_year}_{to_year}.parquet"
    )

    logger.info("Saving data to %s", fname)
    os.makedirs(output_folder, exist_ok=True)
    df.sink_parquet(fname)


@click.command()
@click.option(
    "--from_year", help="Start year to download data", type=click.INT, required=True
)
@click.option(
    "--to_year", help="End year to download data", type=click.INT, required=True
)
@click.option(
    "--output_folder", help="Path to output folder", type=click.STRING, required=True
)
@click.option(
    "--output_folder_raw",
    help="Path to output folder for raw HDF files",
    type=click.STRING,
    required=True,
)
@click.option(
    "--data_type",
    help=f"Data to download: {';'.join(DATA_TYPES)}",
    type=click.STRING,
    required=True,
)
@click.option("--use_cache", is_flag=True, help="Use cached raw data")
def download_ace_data_run(
    from_year: int,
    to_year: int,
    output_folder: str,
    output_folder_raw: str,
    data_type: str,
    use_cache: bool,
) -> None:
    download_ace_data(
        from_year=from_year,
        to_year=to_year,
        output_folder=output_folder,
        output_folder_raw=output_folder_raw,
        data_type=data_type,
        use_cache=use_cache,
    )

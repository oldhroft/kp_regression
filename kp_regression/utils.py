import typing as T

import os
import json
import yaml
import datetime

from uuid import uuid4


def safe_mkdir(name: str) -> None:
    if not os.path.exists(name):
        os.mkdir(name)


def load_yaml(path: str) -> T.Any:
    with open(path, encoding="utf-8", mode="r") as file:
        result = yaml.safe_load(file)
    return result


def load_json(path: str) -> T.Any:
    with open(path, encoding="utf-8", mode="r") as file:
        result = json.load(file)
    return result


def dump_yaml(obj: T.Any, path: str) -> None:
    with open(path, encoding="utf-8", mode="w") as file:
        yaml.safe_dump(obj, file)


def dump_json(obj: T.Any, path: str) -> None:
    with open(path, encoding="utf-8", mode="w") as file:
        json.dump(obj, file)


def add_unique_suffix(name: str, add_date: bool = True, add_uuid: bool = True) -> str:

    if add_date:
        dttm = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if add_uuid:
        uuid = str(uuid4())[:4]

    if add_date and add_uuid:
        return f"{name}_{dttm}_{uuid}"
    elif add_uuid:
        return f"{name}_{uuid}"
    else:
        return f"{name}_{add_date}"

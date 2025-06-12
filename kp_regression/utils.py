import datetime
import json
import os
import typing as T
from uuid import uuid4

import yaml
from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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
        json.dump(obj, file, cls=NumpyEncoder)


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


def serialize_params(model: BaseEstimator) -> T.List[dict]:
    result = []
    if isinstance(model, Pipeline):

        for key, value in model.named_steps.items():
            result.append({"model": key, "params": value.get_params()})
    else:

        result.append({"model": "model", "params": model.get_params()})

    return result

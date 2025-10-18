import typing as T
from dataclasses import dataclass

from kp_regression.utils import load_yaml


@dataclass
class ModelConfig:
    model_name: str
    model_type: str
    model_config: dict
    cv_config: dict
    use_cv: bool
    postprocess_name: str = "Default"

    @classmethod
    def from_dict(cls, dct: dict):
        return cls(**dct)


@dataclass
class DataConfig:
    input_path: T.Union[str, T.Dict[str, str]]
    pipe_name: str
    pipe_params: dict
    split_params: dict
    save_file: bool = False
    save_preds: bool = False
    use_val: bool = False


@dataclass
class Config:
    data_config: DataConfig
    models: T.List[ModelConfig]

    @classmethod
    def from_file(cls, file: str):
        cfg_dict = load_yaml(file)
        if not isinstance(cfg_dict, dict):
            raise ValueError("Config should be a dict!")

        models = cfg_dict["models"]

        if not isinstance(models, list):
            raise ValueError("Models should be a list of models")

        models_list = list(map(ModelConfig.from_dict, models))

        data_config = DataConfig(**cfg_dict["data_config"])

        return Config(data_config=data_config, models=models_list)

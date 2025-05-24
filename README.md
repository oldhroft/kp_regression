# Geomagnetic data forecasting using ML

## How to use

- Install package with poetry (`poetry install`)
- To set up an experiment, create config file ([config description ](https://github.com/oldhroft/kp_regression/blob/main/kp_regression/config.py#L33))
- To run the experiment, use `poetry run --config_path <path-to-config> --exp_folder <folder-to-store-the-experiment>`. [CLI definition](https://github.com/oldhroft/kp_regression/blob/main/kp_regression/run_experiment.py#L27)
- For each model in config folder, a separate folder will be generated, with results.json being file with the results for the model

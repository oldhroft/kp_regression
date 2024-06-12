from kp_regression.run_experiment import run

if __name__ == "__main__":
    run(
        ["--config_path", "config.yaml", "--exp_folder", "./test"],
        standalone_mode=False,
    )

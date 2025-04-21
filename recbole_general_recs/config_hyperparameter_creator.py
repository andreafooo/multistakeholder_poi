import os
import yaml
from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

hyperopt = True


base_config_file_path = "config_base.yaml"
with open(base_config_file_path, "r") as file:
    base_config = yaml.safe_load(file)


datasets = ["foursquaretky_sample"]

models = ["BPR"]


def hyperopt_tune(config_file_path, params_file, output_file):
    hp = HyperTuning(
        objective_function,
        algo="exhaustive",
        early_stop=10,
        max_evals=100,
        params_file=params_file,
        fixed_config_file_list=[config_file_path],
        display_file=None,
    )
    hp.run()
    hp.export_result(output_file=output_file)
    print("best params: ", hp.best_params)
    print("best result: ")
    print(hp.params2result[hp.params2str(hp.best_params)])
    return hp.best_params


for dataset in datasets:
    for model in models:
        config_dir = f"config/{dataset}/{model}"
        os.makedirs(config_dir, exist_ok=True)

        base_config["model"] = model
        base_config["dataset"] = dataset

        config_file_path = os.path.join(config_dir, "config_test.yaml")
        result_file_path = os.path.join(config_dir, "hyper.result")
        with open(config_file_path, "w") as file:
            yaml.dump(base_config, file)
            print("Config file for model", model, "and dataset", dataset, "created")

        if hyperopt:
            best_params = hyperopt_tune(
                config_file_path, "hyper.test", result_file_path
            )
            base_config["learning_rate"] = best_params["learning_rate"]
            base_config["embedding_size"] = best_params["embedding_size"]
            base_config["train_batch_size"] = best_params["train_batch_size"]
            with open(config_file_path, "w") as file:
                yaml.dump(base_config, file)
                print("Updated config file with best hyperparameters")

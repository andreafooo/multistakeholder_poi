import ast
import copy
import os
import sys
import types
import yaml

# recbole.model.general_recommender's __init__ eagerly imports LDiffRec, which
# imports kmeans_pytorch at module level. get_model() (used to resolve BPR/NeuMF
# by name) triggers that whole package import, so without kmeans_pytorch installed
# every run crashes even though we never touch LDiffRec. Stub it out instead of
# requiring a hand-edit to the installed recbole package on every machine/venv.
if "kmeans_pytorch" not in sys.modules:
    try:
        import kmeans_pytorch  # noqa: F401
    except ImportError:
        _kmeans_pytorch_stub = types.ModuleType("kmeans_pytorch")
        _kmeans_pytorch_stub.kmeans = None
        sys.modules["kmeans_pytorch"] = _kmeans_pytorch_stub

from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import datasets_for_recbole, models_for_recbole

# hyperopt's rand.suggest (used by algo="random") builds its RNG via
# np.random.default_rng(), which lacks .randint() (only .integers()); hyperopt's
# pyll "randint" op (backing every hp.choice) still calls rng.randint(low, high, size),
# so "random" search crashes immediately without this. Generator is an immutable
# extension type, so patch the pyll op implementation instead of the class.
from hyperopt.pyll.base import scope as _hyperopt_scope


def _randint_via_integers(low, high=None, rng=None, size=()):
    # exhaustive_search still passes a legacy RandomState (has .randint, no .integers);
    # rand.suggest passes a new-style Generator (has .integers, no .randint).
    if hasattr(rng, "randint"):
        return rng.randint(low, high, size)
    return rng.integers(low, high, size)


_hyperopt_scope._impls["randint"] = _randint_via_integers

hyperopt = True
hyperopt_algo = "random"  # "exhaustive" evaluates the full grid; "random" samples max_evals points from it
hyperopt_max_evals = 60


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

base_config_file_path = os.path.join(SCRIPT_DIR, "config_base.yaml")
with open(base_config_file_path, "r") as file:
    base_config = yaml.safe_load(file)


datasets = datasets_for_recbole

# config_base.yaml's data_path is left as a machine-specific absolute path for
# convenience when editing it by hand. A relative path here is fragile (it depends
# on whatever directory the training script happens to be invoked from - both
# repo-root-relative and recbole_general_recs-relative variants have broken before
# depending on invocation site), so we never persist any data_path into the
# committed config_test.yaml files. Instead, the real absolute path is injected
# only at runtime via a local, gitignored override file layered on top via
# fixed_config_file_list, so it never depends on cwd and never leaks into git.
base_config.pop("data_path", None)

data_path_override_file = os.path.join(SCRIPT_DIR, ".data_path_override.yaml")
with open(data_path_override_file, "w") as file:
    yaml.dump({"data_path": os.path.join(SCRIPT_DIR, "dataset") + os.sep}, file)

# each general recommendation model gets its own hyperparameter search space,
# since the tunable parameter names differ per model (e.g. BPR's "embedding_size"
# vs NeuMF's "mf_embedding_size"/"mlp_embedding_size")
model_hyperparams_files = {
    "BPR": os.path.join(SCRIPT_DIR, "hyper.test"),
    "NeuMF": os.path.join(SCRIPT_DIR, "hyper_neumf.test"),
}


def hyperopt_tune(config_file_path, params_file, output_file):
    hp = HyperTuning(
        objective_function,
        algo=hyperopt_algo,
        early_stop=20,
        max_evals=hyperopt_max_evals,
        params_file=params_file,
        fixed_config_file_list=[config_file_path, data_path_override_file],
        display_file=None,
    )
    hp.run()
    hp.export_result(output_file=output_file)
    print("best params: ", hp.best_params)
    print("best result: ")
    print(hp.params2result[hp.params2str(hp.best_params)])
    return hp.best_params


for dataset in datasets_for_recbole:
    for model in models_for_recbole:
        config_dir = os.path.join(SCRIPT_DIR, "config", dataset, model)
        os.makedirs(config_dir, exist_ok=True)

        model_config = copy.deepcopy(base_config)
        model_config["model"] = model
        model_config["dataset"] = dataset

        config_file_path = os.path.join(config_dir, "config_test.yaml")
        result_file_path = os.path.join(config_dir, "hyper.result")
        with open(config_file_path, "w") as file:
            yaml.dump(model_config, file)
            print("Config file for model", model, "and dataset", dataset, "created")

        if hyperopt:
            params_file = model_hyperparams_files[model]
            best_params = hyperopt_tune(
                config_file_path, params_file, result_file_path
            )
            # RecBole's HyperTuning encodes list-valued choices (e.g. mlp_hidden_size)
            # as their string repr, since hyperopt's hp.choice requires hashable
            # options and a raw list isn't hashable. Decode them back before writing,
            # otherwise NeuMF later fails concatenating a list with this raw string.
            for key, value in best_params.items():
                if isinstance(value, str):
                    try:
                        best_params[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        pass
            model_config.update(best_params)
            with open(config_file_path, "w") as file:
                yaml.dump(model_config, file)
                print(
                    f"Updated config file with best hyperparameters in dir {config_dir}"
                )

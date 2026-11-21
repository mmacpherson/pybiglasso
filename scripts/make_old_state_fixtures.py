"""Fitted paths and estimators pickled by an earlier pybiglasso.

tests/fixtures/old_state/ holds three of the files this wrote, for the test
that refuses fitted states from before centered prediction state was
recorded. They were made with a release-profile wheel built from commit
b88db8b, installed alone: python -I make_old_state_fixtures.py --wheel WHEEL.
This copy differs from the one run only in import order.
"""

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import pickletools
import platform
import re
import warnings
import zipfile
import zlib
from pathlib import Path, PurePath

import joblib
import numpy as np
import pybiglasso._rust as native
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

import pybiglasso.fit as fit_module
from pybiglasso import biglasso
from pybiglasso.path import BiglassoPath
from pybiglasso.sklearn import (
    BiglassoClassifier,
    BiglassoClassifierCV,
    BiglassoRegressor,
    BiglassoRegressorCV,
)

COMMIT = "b88db8b7e9462110492ac4ca86a2a2aa3b62d11a"
SEED = 820260
PROTOCOL = 4
JOBLIB_COMPRESSION = ("zlib", 3)
ABSOLUTE_PATH = re.compile(
    rb"(?<![A-Za-z0-9_])(?:/[A-Za-z0-9_.-]+/|[A-Za-z]:[\\/]|\\\\[A-Za-z0-9_.-]+\\)"
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_wheel(wheel):
    package = Path(native.__file__).parent
    assert Path(fit_module.__file__).parent == package
    with zipfile.ZipFile(wheel) as archive:
        for name in ("fit.py", "path.py", "sklearn.py", Path(native.__file__).name):
            assert (package / name).read_bytes() == archive.read(f"pybiglasso/{name}")


def make_inputs(root):
    rng = np.random.default_rng(SEED)
    train = np.asfortranarray(rng.standard_normal((192, 6)))
    probe = np.asfortranarray(rng.standard_normal((8, 6)))
    shift = np.array([3.0, -2.0, 0.0, 1.0, -1.0, 0.5])
    train += shift
    probe += shift
    gaussian = 1.5 + train @ np.array([1.0, -0.8, 0.0, 0.5, 0.0, 0.0])
    gaussian += 0.5 * rng.standard_normal(len(train))
    logits = (train - shift) @ np.array([0.9, -0.7, 0.0, 0.6, 0.0, 0.0])
    probability = 1 / (1 + np.exp(-logits))
    binomial = (rng.random(len(train)) < probability).astype(np.int64)
    arrays = {
        "X_train": train,
        "X_predict": probe,
        "y_gaussian": gaussian,
        "y_binomial": binomial,
    }
    for key, value in arrays.items():
        np.save(root / "inputs" / f"{key}.npy", value, allow_pickle=False)
    return arrays


def predictions(model, X, methods):
    result = {}
    for method in methods:
        values = getattr(model, method)(X)
        result[method] = {
            "dtype": values.dtype.name,
            "shape": list(values.shape),
            "values": values.tolist(),
        }
    return result


def reject_unfitted(model, X):
    try:
        model.predict(X)
    except NotFittedError:
        return
    raise AssertionError("An unfitted estimator unexpectedly predicted")


def audit_serialization(path):
    contents = path.read_bytes()
    if path.suffix == ".joblib":
        contents = zlib.decompress(contents)
    else:
        # Check literal string operands in the pickle as well as its raw bytes.
        for op, argument, _ in pickletools.genops(contents):
            if isinstance(argument, str):
                assert not ABSOLUTE_PATH.search(argument.encode()), (path.name, op.name)
    assert contents[:2] == bytes((0x80, PROTOCOL))
    assert not ABSOLUTE_PATH.search(contents), path.name


def audit_state(value, seen):
    # Inspect strings and path-like values in the restored state, including
    # joblib's container fields whose binary array buffers interrupt pickle opcodes.
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, (str, bytes, PurePath)):
        raw = value if isinstance(value, bytes) else str(value).encode()
        assert not raw.startswith((b"/", b"\\"))
        assert not ABSOLUTE_PATH.search(raw)
    elif isinstance(value, dict):
        for key, item in value.items():
            audit_state(key, seen)
            audit_state(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            audit_state(item, seen)
    elif isinstance(value, np.ndarray) and value.dtype.kind in "OUST":
        for item in value.flat:
            audit_state(item, seen)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        audit_state(vars(value), seen)


def save_model(root, name, model, probe, description, expected, train, y):
    methods = description["prediction_methods"]
    files = []
    for suffix in ("pkl", "joblib"):
        relative = f"models/{name}.{suffix}"
        path = root / relative
        if suffix == "pkl":
            path.write_bytes(pickle.dumps(model, protocol=PROTOCOL))
            restored = pickle.loads(path.read_bytes())
        else:
            joblib.dump(model, path, compress=JOBLIB_COMPRESSION, protocol=PROTOCOL)
            restored = joblib.load(path)
        audit_serialization(path)
        audit_state(restored, set())
        assert type(restored) is type(model)
        if description["state"] == "unfitted":
            assert restored.get_params(deep=False) == model.get_params(deep=False)
            reject_unfitted(restored, probe)
            restored.fit(train, y)
        if not isinstance(restored, BiglassoPath):
            check_is_fitted(restored)
        assert predictions(restored, probe, methods) == expected, relative
        files.append(relative)
    return {**description, "files": files, "expected_key": name}


def generate(root, wheel):
    verify_wheel(wheel)
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    script = Path(__file__).resolve()
    if script != (root / "generate.py").resolve():
        (root / "generate.py").write_bytes(script.read_bytes())
    arrays = make_inputs(root)
    train, probe = arrays["X_train"], arrays["X_predict"]
    expected = {}
    objects = {}

    def record(name, model, family, state, methods, options, expected_after_fit=None):
        y = arrays[f"y_{family}"]
        description = {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "family": family,
            "state": state,
            "prediction_methods": methods,
            "prediction_input": "inputs/X_predict.npy",
            "fit_inputs": ["inputs/X_train.npy", f"inputs/y_{family}.npy"],
            **options,
        }
        if state == "fitted":
            target = predictions(model, probe, methods)
            expected[name] = {"outputs": target}
        else:
            reject_unfitted(model, probe)
            target = expected[expected_after_fit]["outputs"]
            expected[name] = {
                "before_fit": {"predict": "sklearn.exceptions.NotFittedError"},
                "after_fit_expected_key": expected_after_fit,
            }
        objects[name] = save_model(
            root, name, model, probe, description, target, train, y
        )

    for family in ("gaussian", "binomial"):
        methods = ["predict", "decision_function"]
        if family == "binomial":
            methods.append("predict_proba")
        for compatible in (False, True):
            name = f"path_{family}_r_compatible_{str(compatible).lower()}"
            options = {
                "family": family,
                "r_compatible": compatible,
                "nlambda": 9,
                "lambda_min": 0.03,
                "n_threads": 1,
            }
            model = biglasso(train, arrays[f"y_{family}"], **options)
            assert len(model.lambdas) == options["nlambda"]
            record(name, model, family, "fitted", methods, {"fit_kwargs": options})

    for estimator, family in (
        (BiglassoClassifier, "binomial"),
        (BiglassoClassifierCV, "binomial"),
        (BiglassoRegressor, "gaussian"),
        (BiglassoRegressorCV, "gaussian"),
    ):
        methods = (
            ["predict", "decision_function", "predict_proba"]
            if family == "binomial"
            else ["predict"]
        )
        options = (
            {"n_alphas": 9, "cv": 3, "n_jobs": 1}
            if estimator.__name__.endswith("CV")
            else {"alpha": 0.05}
        )
        fitted_name = f"{estimator.__name__}_fitted"
        fitted = estimator(**options).fit(train, arrays[f"y_{family}"])
        record(
            fitted_name,
            fitted,
            family,
            "fitted",
            methods,
            {"constructor_kwargs": options},
        )
        record(
            f"{estimator.__name__}_unfitted",
            estimator(**options),
            family,
            "unfitted",
            methods,
            {"constructor_kwargs": options},
            expected_after_fit=fitted_name,
        )

    write_json(root / "expected.json", expected)
    inventory = {}
    for name, entry in objects.items():
        for relative in entry["files"]:
            inventory[relative] = {"role": "serialized object", "object": name}
    for key in arrays:
        inventory[f"inputs/{key}.npy"] = {"role": f"NumPy fixture array {key}"}
    inventory["expected.json"] = {
        "role": "old-wheel prediction results and unfitted behavior, keyed by object"
    }
    inventory["generate.py"] = {"role": "generator and round-trip verification"}
    for relative, entry in inventory.items():
        content = (root / relative).read_bytes()
        entry.update(bytes=len(content), sha256=sha256(content))
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "pybiglasso",
            "numpy",
            "scipy",
            "scikit-learn",
            "joblib",
            "narwhals",
            "cloudpickle",
            "threadpoolctl",
        )
    }
    manifest = {
        "source_commit": COMMIT,
        "wheel_build": {
            "source": "git archive of source_commit",
            "command": "maturin build --release --locked",
            "maturin": "1.11.5",
            "rustc": "1.98.0",
        },
        "wheel_sha256": sha256(wheel.read_bytes()),
        "versions": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            **versions,
        },
        "serialization": {
            "pickle_protocol": PROTOCOL,
            "joblib_pickle_protocol": PROTOCOL,
            "joblib_compression": list(JOBLIB_COMPRESSION),
        },
        "input_seed": SEED,
        "numerical_library_thread_limit": 1,
        "expected_file": "expected.json",
        "objects": objects,
        "files": inventory,
        "manifest_self_hash": "Excluded from the manifest's SHA-256 inventory.",
    }
    write_json(root / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel",
        required=True,
        type=Path,
        help="archived source wheel already installed in this environment",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error")
        generate(args.output, args.wheel)

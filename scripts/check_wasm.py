"""Run the browser binding against the existing, independently generated R grid.

Build web/pkg first. The selected cases need no row subsets or linear grids;
they include early truncation, constant columns, penalties and max_iter. Input
digests must match the stored R fixtures. No pybiglasso fit supplies an expected
value, and the comparison uses the existing R policy, not a new tolerance.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from support import rcompare  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", choices=["node", "chromium", "firefox", "webkit"], default="node"
    )
    parser.add_argument(
        "--native-bits",
        action="store_true",
        help="also check same-input identity with the running native backend",
    )
    parser.add_argument(
        "--container", help="run the JS/browser side in this Playwright Docker image"
    )
    args = parser.parse_args()
    fixtures = rcompare.load_fixtures()
    cases, payload, expected, native = [], [], [], []
    with tempfile.TemporaryDirectory() as temp:
        for case in rcompare.fast_grid():
            if case.rows != "all" or not case.lambda_log_scale:
                continue
            digest, lambdas, reference = fixtures[case.name]
            data = rcompare.make_case_data(case, Path(temp) / case.name)
            assert rcompare.data_digest(data) == digest, case.name
            options = dict(
                family="binomial",
                alpha=case.alpha,
                nlambda=case.nlambda,
                eps=rcompare.EPS,
                lambda_min=case.resolved_lambda_min,
                max_iter=case.max_iter,
                r_compatible=True,
            )
            if lambdas is not None:
                options["lambdas"] = np.asarray(lambdas).tolist()
            if data["penalty_factor"] is not None:
                options["penalty_factor"] = data["penalty_factor"].tolist()
            if case.dfmax is not None:
                options["dfmax"] = case.dfmax
            cases.append(case)
            expected.append(reference)
            if args.native_bits:
                native.append(rcompare.fit_py(case, data, lambdas))
            payload.append(
                dict(
                    X=np.asarray(data["X"], dtype=float).tolist(),
                    y=data["y"].tolist(),
                    options=options,
                )
            )
        command = ["node", str(ROOT / "web/scripts/run-fixtures.mjs"), args.engine]
        if args.container:
            command = [
                "docker",
                "run",
                "--rm",
                "--init",
                "--ipc=host",
                "-i",
                "-v",
                f"{ROOT}:/work",
                "-w",
                "/work",
                args.container,
                "node",
                "web/scripts/run-fixtures.mjs",
                args.engine,
            ]
        result = subprocess.run(
            command,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        )
    for index, (case, actual, reference) in enumerate(
        zip(cases, json.loads(result.stdout), expected, strict=True)
    ):
        coef = np.zeros((case.ncol + 1, len(actual["lambdas"])))
        coef[0] = actual["intercept"]
        for k, (indices, values) in enumerate(
            zip(actual["indices"], actual["coefficients"], strict=True)
        ):
            coef[1 + np.asarray(indices, dtype=int), k] = values
        for key, got in [
            ("center", actual["center"]),
            ("scale", actual["scale"]),
            ("lambda_", actual["lambdas"]),
            ("loss", np.asarray(actual["deviance"]) / 2),
            ("beta", coef),
        ]:
            np.testing.assert_allclose(
                got,
                reference[key],
                rtol=rcompare.RTOL,
                atol=rcompare.ATOL,
                err_msg=f"{case.name}: {key}",
            )
            if args.native_bits:
                a = np.asarray(got, dtype=np.float64)
                b = np.asarray(native[index][key], dtype=np.float64)
                assert a.shape == b.shape and a.tobytes() == b.tobytes(), (
                    f"{args.engine} vs native bits: {case.name}: {key}"
                )
        np.testing.assert_array_equal(
            actual["iterations"], reference["iter"], err_msg=case.name
        )
        np.testing.assert_array_equal(
            actual["n_rejected"], reference["rejections"], err_msg=case.name
        )
    print(
        f"Wasm ({args.engine}) matches stored R output on {len(cases)} cases, including iterations and rejections."
    )
    if args.native_bits:
        print("Same-input numerical outputs are also bit-identical to native Rust.")


if __name__ == "__main__":
    main()

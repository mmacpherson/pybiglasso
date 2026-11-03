"""Exact-rational fast/safe/dispatch audit at every range-guard boundary."""

import importlib.util
import json
import math
import random
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

library = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("_rust", library)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
low, high = math.ldexp(1.0, -240), math.ldexp(1.0, 240)
values = [
    0.0,
    -0.0,
    math.nextafter(low, 0.0),
    low,
    math.nextafter(low, math.inf),
    math.nextafter(high, 0.0),
    high,
    math.nextafter(high, math.inf),
]
cases = [[], [0.0], [-0.0, 0.0]]
for value in values:
    for sign in [-1.0, 1.0]:
        for n in [1, 2, 7, 8, 9, 63, 64, 65, 2047, 2048, 2049, 50000]:
            cases.append([sign * value] * n)
        cases.extend(
            [
                [1.0, sign * value, -0.0],
                [sign * value, low, high],
                [low, high, sign * value],
            ]
        )
# Keep an accepted vector on each side of the fallback transition: changing
# one last/middle/first component alone must select the safe path.
for inside, outside in [
    (low, math.nextafter(low, 0.0)),
    (high, math.nextafter(high, math.inf)),
]:
    for index in [0, 4, 8]:
        v = [inside] * 9
        v[index] = outside
        cases.append(v)
cases += [[1e-200], [math.ulp(0.0)], [1e-300, 1e300], [float("nan")], [float("inf")]]
rng = random.Random(46109)
for _ in range(300):
    cases.append(
        [
            math.ldexp(
                rng.uniform(-1.0, 1.0), rng.choice([-241, -240, -239, 0, 239, 240, 241])
            )
            for _ in range(rng.choice([1, 7, 8, 9, 31]))
        ]
    )
fast_count = safe_count = none_count = 0
for vector in cases:
    fast, safe, selected = native._vep_norm_paths(vector)
    accepted = all(value == 0.0 or low <= abs(value) <= high for value in vector)
    assert (fast is not None) == accepted, ("guard routing", vector, fast)
    assert selected == (fast if accepted else safe), ("dispatch", vector)
    if all(math.isfinite(value) for value in vector):
        square = sum(
            (Fraction(value) ** 2 * count for value, count in Counter(vector).items()),
            Fraction(0),
        )
        for branch, result in [("fast", fast), ("safe", safe), ("selected", selected)]:
            if result is not None:
                assert math.isfinite(result) and result >= 0.0
                assert Fraction(result) ** 2 >= square, (
                    "underbound",
                    branch,
                    vector,
                    result,
                )
    if accepted:
        fast_count += 1
    elif selected is not None:
        safe_count += 1
    else:
        none_count += 1
# Guard the transformed input, not an earlier raw vector. These pairs force
# opposite routing decisions before and after centering/differencing.
raw_large = [2 * high, math.nextafter(2 * high, math.inf)]
quantum = raw_large[1] - raw_large[0]
transformed_routes = [
    ([high, -high], True),
    ([0.0, math.nextafter(2 * high, math.inf)], False),
    (raw_large, False),
    ([(v - raw_large[0]) / quantum for v in raw_large], True),
    ([quantum, 0.0], True),
]
for vector, expected_fast in transformed_routes:
    fast, safe, selected = native._vep_norm_paths(vector)
    assert (fast is not None) == expected_fast
    assert selected == (fast if expected_fast else safe)
    exact_square = sum((Fraction(v) ** 2 for v in vector), Fraction(0))
    assert selected is not None and Fraction(selected) ** 2 >= exact_square
score_enclosures = score_fallbacks = 0
for n in [7, 8, 9]:
    for magnitude in values + [1e-200]:
        for center in [0.0, 1.0, -1.0, 100.0, math.nextafter(100.0, math.inf)]:
            xx = [
                center if i % 3 == 0 else center + (-1.0) ** i * magnitude
                for i in range(n)
            ]
            for residual_scale in [low, 1.0, high, 1e-200]:
                ra = [0.0] * n
                rb = [(-1.0) ** i * residual_scale for i in range(n)]
                result = native._vep_bound(xx, ra, rb, center, 1.0, n)
                if result is None:
                    score_fallbacks += 1
                    continue
                qa, qb, radius, na, nb, delta, coefficient, ea, eb = result
                if abs(center) > 100.0:
                    coefficients = [Fraction((v - center) / 1.0) / n for v in xx]
                else:
                    coefficients = [
                        (Fraction(v) - Fraction(center)) / Fraction(float(n))
                        for v in xx
                    ]
                exact_a = sum(
                    (v * Fraction(r) for v, r in zip(coefficients, ra, strict=True)),
                    Fraction(0),
                )
                exact_b = sum(
                    (v * Fraction(r) for v, r in zip(coefficients, rb, strict=True)),
                    Fraction(0),
                )
                assert Fraction(coefficient) ** 2 >= sum(
                    (v * v for v in coefficients), Fraction(0)
                )
                assert abs(Fraction(qa) - exact_a) <= Fraction(ea)
                assert abs(Fraction(qb) - exact_b) <= Fraction(eb)
                assert abs(Fraction(qb) - Fraction(qa)) <= Fraction(radius)
                score_enclosures += 1
print(
    json.dumps(
        dict(
            cases=len(cases),
            transformed_guard_cases=len(transformed_routes),
            score_enclosures=score_enclosures,
            score_safe_fallbacks=score_fallbacks,
            fast_enclosures=fast_count,
            guarded_safe_fallbacks=safe_count,
            invalid_safe_fallbacks=none_count,
            expected="exact Fraction sum of squares; no pybiglasso-generated expectations",
        )
    )
)

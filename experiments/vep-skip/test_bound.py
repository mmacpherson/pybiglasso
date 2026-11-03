"""Exact-rational audit of the experimental floating point enclosures."""

import importlib.util
import json
import math
import random
import sys
from fractions import Fraction as F
from pathlib import Path

library = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("_rust", library)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
norm_cases = [
    [0.0],
    [1e-200],
    [math.ulp(0.0)],
    [math.ulp(0.0), math.ulp(0.0)],
    [1e-300, 1e300],
    [1e200],
    [1e308],
    [float.fromhex("0x1.fffffffffffffp+1023")],
    [float("nan")],
    [float("inf")],
    [],
]
rng = random.Random(84623)
for _ in range(200):
    norm_cases.append(
        [
            math.ldexp(rng.uniform(-1, 1), rng.randrange(-1073, 1024))
            for _ in range(rng.choice([1, 4, 9, 17]))
        ]
    )
valid_norm = 0
fallback_norm = 0
for v in norm_cases:
    q = m._vep_norm(v)
    if q is None:
        fallback_norm += 1
        continue
    assert math.isfinite(q) and q >= 0
    assert F(q) ** 2 >= sum((F(x) ** 2 for x in v), F(0)), ("norm underbound", v, q)
    valid_norm += 1
cases = []
x = [-1.0, 1.0, -1.0, 1.0]
cases.append((x, [0.0] * 4, [1e-200 * t for t in x], 0.0, 1.0, 4))
for e in [-1073, -1022, -900, -665, -500, -100, 0, 100, 500, 900, 1000]:
    for re in [-1000, -665, -500, -100, 0, 100, 500, 900, 1000]:
        xx = [math.ldexp(t, e) for t in x]
        ra = [0.0] * 4
        rb = [math.ldexp(t, re) for t in x]
        cases.append((xx, ra, rb, 0.0, 1.0, 4))
for _ in range(300):
    n = rng.choice([1, 4, 9, 17])
    base = math.ldexp(1.0, rng.randrange(-900, 901))
    rscale = math.ldexp(1.0, rng.randrange(-900, 901))
    xx = [base * rng.uniform(-1, 1) for _ in range(n)]
    ra = [rscale * rng.uniform(-1, 1) for _ in range(n)]
    rb = [v * (1 + math.ldexp(rng.uniform(-1, 1), -20)) for v in ra]
    cases.append((xx, ra, rb, 0.0, math.ldexp(1.0, rng.randrange(-900, 901)), n))
for c in [1e8, 1e16, 1e100]:
    xx = [
        math.nextafter(c, -math.inf),
        c,
        math.nextafter(c, math.inf),
        math.nextafter(math.nextafter(c, math.inf), math.inf),
    ]
    for scale in [1.0, math.ulp(c), 1e-200]:
        cases.append(
            (xx, [1.0, -1.0, 1.0, -1.0], [1.0, -1.0, 1.0, -1.0 + 2**-40], c, scale, 4)
        )
cases.extend(
    [
        (x, [0.0] * 4, x, 0.0, 0.0, 4),
        (x, [0.0] * 4, x, float("nan"), 1.0, 4),
        (x, [0.0] * 4, x, 0.0, 1.0, 2**54),
    ]
)
valid = 0
fallback = 0
decisions = 0
for xx, ra, rb, c, s, n in cases:
    out = m._vep_bound(xx, ra, rb, c, s, n)
    if out is None:
        fallback += 1
        continue
    qa, qb, radius, na, nb, delta, coefficient, ea, eb = out
    assert all(math.isfinite(v) and v >= 0 for v in out[2:]), out
    assert abs(F(qb) - F(qa)) <= F(radius), (
        "score delta underbound",
        xx,
        ra,
        rb,
        c,
        s,
        n,
        out,
    )
    if abs(c) > 100 * s:
        z = [(v - c) / s for v in xx]

        def exact(r, z=z, n=n):
            return sum((F(v) * F(t) for v, t in zip(z, r, strict=True)), F(0)) / F(
                float(n)
            )
    else:
        d = s * float(n)

        def exact(r, xx=xx, c=c, d=d):
            return sum(
                ((F(v) - F(c)) * F(t) for v, t in zip(xx, r, strict=True)), F(0)
            ) / F(d)

    assert abs(F(qa) - exact(ra)) <= F(ea), ("anchor score error underbound", out)
    assert abs(F(qb) - exact(rb)) <= F(eb), ("current score error underbound", out)
    assert F(delta) ** 2 >= sum(
        ((F(a) - F(b)) ** 2 for a, b in zip(ra, rb, strict=True)), F(0)
    )
    for threshold in [
        0.0,
        abs(qb),
        math.nextafter(abs(qb), 0.0),
        math.nextafter(abs(qb), math.inf),
        abs(qb) * 0.5,
        abs(qb) * 2,
    ]:
        decision = m._vep_decide(qa, radius, threshold)
        if decision is not None:
            assert decision == (abs(qb) > threshold), (
                "wrong certificate",
                out,
                threshold,
                decision,
            )
            decisions += 1
    valid += 1
adversarial = m._vep_bound(x, [0.0] * 4, [1e-200 * t for t in x], 0.0, 1.0, 4)
assert m._vep_decide(adversarial[0], adversarial[2], 5e-201) is not False
print(
    json.dumps(
        dict(
            norm_enclosures_checked=valid_norm,
            norm_safe_fallbacks=fallback_norm,
            score_enclosures_checked=valid,
            score_safe_fallbacks=fallback,
            certified_near_threshold_decisions=decisions,
            astra_counterexample=dict(
                score=adversarial[1],
                radius=adversarial[2],
                decision=m._vep_decide(adversarial[0], adversarial[2], 5e-201),
            ),
        )
    )
)

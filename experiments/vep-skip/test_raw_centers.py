"""Astra round-1 rational cases; only binary selection is parameterized."""

import ctypes
import importlib.util
import json
import math
import random
import sys
from fractions import Fraction as F

p = sys.argv[1]
s = importlib.util.spec_from_file_location("_rust", p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
rng = random.Random(92)
valid = 0
fallback = 0
for k in range(1200):
    n = rng.choice([4, 8, 9, 17, 33, 65])
    scale = math.ldexp(1.0, rng.randint(-900, 900))
    ratio = rng.choice(
        [0.5, 3.0, 99.0, 100.0, math.nextafter(100.0, math.inf), 101.0, 1000.0]
    )
    center = scale * ratio * (-1 if k % 2 else 1)
    x = [center + scale * rng.uniform(-2, 2) for _ in range(n)]
    magnitude = math.ldexp(1.0, rng.randint(-1000, 900))
    a = [magnitude * rng.uniform(-1, 1) for _ in range(n)]
    b = [v * (1 + rng.uniform(-0.1, 0.1)) for v in a]
    out = m._vep_bound(x, a, b, center, scale, n)
    if out is None:
        fallback += 1
        continue
    qa, qb, rad, na, nb, dist, cn, ea, eb = out
    if abs(center) > 100 * scale:
        z = [F((v - center) / scale) for v in x]
        d = F(float(n))
    else:
        z = [F(v) - F(center) for v in x]
        d = F(scale * n)

    def exact(r, z=z, d=d):
        return sum((v * F(t) for v, t in zip(z, r, strict=True)), F(0)) / d

    assert abs(F(qa) - exact(a)) <= F(ea), (k, "anchor")
    assert abs(F(qb) - exact(b)) <= F(eb), (k, "current")
    assert abs(F(qb) - F(qa)) <= F(rad), (k, "radius")
    assert F(cn) ** 2 >= sum((v * v for v in z), F(0)) / (d * d), (k, "coefficient")
    valid += 1
libc = ctypes.CDLL(None)
old = libc.fegetround()
modes = {}
try:
    for name, value in [
        ("nearest", 0),
        ("down", 0x400),
        ("up", 0x800),
        ("zero", 0xC00),
    ]:
        assert libc.fesetround(value) == 0
        q = m._vep_norm([1.0, 2.0])
        modes[name] = q is None
finally:
    libc.fesetround(old)
assert modes == {"nearest": False, "down": True, "up": True, "zero": True}, modes
print(
    json.dumps(
        dict(
            raw_and_fallback_valid=valid,
            safe_fallbacks=fallback,
            directed_rounding_rejected=modes,
        )
    )
)

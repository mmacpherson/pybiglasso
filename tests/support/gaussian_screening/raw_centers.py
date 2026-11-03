def audit(native):
    """Retained rational cases with nonzero centers and fallback boundaries."""

    import math
    import random
    from fractions import Fraction as F

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
        out = native.bound(x, a, b, center, scale, n)
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
    assert valid > 0 and fallback > 0

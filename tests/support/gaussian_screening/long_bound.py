def audit(native):
    """Retained rational cases with nonzero centers and fallback boundaries."""

    import math
    import random
    from fractions import Fraction as F

    rng = random.Random(11)
    for n in (7, 8, 9, 2047, 2048, 2049, 16385):
        for c in (37.0, 101.0):
            scale = 0.5
            x = [c + rng.randint(-16, 16) / 8 for _ in range(n)]
            a = [rng.randint(-32, 32) / 8 for _ in range(n)]
            b = [math.nextafter(t, math.inf) for t in a]
            qa, qb, r, na, nb, d, cn, ea, eb = native.bound(x, a, b, c, scale, n)
            if abs(c) > 100 * scale:
                z = [F((t - c) / scale) for t in x]
                den = F(float(n))
            else:
                z = [F(t) - F(c) for t in x]
                den = F(scale * n)

            def exact(v, z=z, den=den):
                return sum((t * F(w) for t, w in zip(z, v, strict=True)), F()) / den

            assert abs(F(qa) - exact(a)) <= F(ea)
            assert abs(F(qb) - exact(b)) <= F(eb)
            assert abs(F(qb) - F(qa)) <= F(r)
    print("14 long/lane-boundary rational score checks passed")

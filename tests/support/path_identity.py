"""Framed path identity for invariance checks, including sparse storage."""

import hashlib
import json
import struct

import numpy as np


def path_digest(p):
    h = hashlib.sha256(b"gaussian-path-digest-v2\0")
    fields = {
        "coef.dense": p.coef.toarray(),
        "coef.format": p.coef.format,
        "coef.shape": p.coef.shape,
        "coef.data": p.coef.data,
        "coef.indices": p.coef.indices,
        "coef.indptr": p.coef.indptr,
        "intercept": p.intercept,
        "lambdas": p.lambdas,
        "deviance": p.deviance,
        "iterations": p.iterations,
        "n_rejected": p.n_rejected,
        "center": p.center,
        "scale": p.scale,
        "penalty_factor": p.penalty_factor,
        "alpha": p.alpha,
    }
    for name, value in fields.items():
        a = np.asarray(value)
        frame = json.dumps([name, a.dtype.str, a.shape]).encode()
        payload = a.tobytes()
        h.update(struct.pack("<QQ", len(frame), len(payload)))
        h.update(frame)
        h.update(payload)
    return h.hexdigest()

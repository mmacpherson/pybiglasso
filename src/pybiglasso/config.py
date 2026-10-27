"""pybiglasso's settings, read from the environment once, at import.

Each field of `Settings` can be set by an environment variable named for it:
PYBIGLASSO_ and the field's name in upper case (PYBIGLASSO_BOUNDSCHECK). An
argument passed to a function overrides the setting, which overrides the
default.
"""

import dataclasses
import os

BACKENDS = ("rust",)
PREFIX = "PYBIGLASSO_"
SWITCH = {"1": True, "true": True, "on": True, "0": False, "false": False, "off": False}


def check_backend(backend):
    if backend == "numba":
        raise ValueError(
            "The numba backend was removed in pybiglasso 0.4.0; "
            'use backend="rust" or omit backend.'
        )
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}; got {backend!r}.")
    return backend


def check_switch(name, value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in SWITCH:
        return SWITCH[value.strip().lower()]
    raise ValueError(
        f"{name} must be on or off (1, true, on / 0, false, off); got {value!r}."
    )


@dataclasses.dataclass(frozen=True)
class Settings:
    # Check every index into an array against its bounds. Inputs are checked
    # before a fit either way, so this guards against a bug in pybiglasso
    # itself, at some cost in speed; results are the same with it on or off.
    boundscheck: bool = True

    def __post_init__(self):
        object.__setattr__(
            self, "boundscheck", check_switch("boundscheck", self.boundscheck)
        )

    @classmethod
    def from_env(cls, environ=os.environ):
        # No backend selection remains. Diagnose old configurations rather
        # than silently running a different implementation after upgrading.
        if PREFIX + "BACKEND" in environ:
            try:
                check_backend(environ[PREFIX + "BACKEND"])
            except ValueError as e:
                raise ValueError(
                    f"{e} (from the environment: PYBIGLASSO_BACKEND)"
                ) from None
        values = {}
        for field in dataclasses.fields(cls):
            name = PREFIX + field.name.upper()
            if name in environ:
                values[field.name] = environ[name]
        try:
            return cls(**values)
        except ValueError as e:
            set_here = ", ".join(PREFIX + k.upper() for k in values)
            raise ValueError(f"{e} (from the environment: {set_here})") from None


settings = Settings.from_env()

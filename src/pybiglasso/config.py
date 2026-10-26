"""pybiglasso's settings, read from the environment once, at import.

Each field of `Settings` can be set by an environment variable named for it:
PYBIGLASSO_ and the field's name in upper case (PYBIGLASSO_BACKEND). An
argument passed to a function overrides the setting, which overrides the
default.
"""

import dataclasses
import os

BACKENDS = ("numba", "rust")
PREFIX = "PYBIGLASSO_"
SWITCH = {"1": True, "true": True, "on": True, "0": False, "false": False, "off": False}


def check_backend(backend):
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
    # "numba" is deprecated, an optional extra, and goes in 0.4.0
    backend: str = "rust"
    # Check every index into an array against its bounds. Inputs are checked
    # before a fit either way, so this guards against a bug in pybiglasso
    # itself, at some cost in speed; results are the same with it on or off.
    # The numba backend reads it once, at import.
    boundscheck: bool = True

    def __post_init__(self):
        check_backend(self.backend)
        object.__setattr__(
            self, "boundscheck", check_switch("boundscheck", self.boundscheck)
        )

    @classmethod
    def from_env(cls, environ=os.environ):
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

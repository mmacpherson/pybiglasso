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


def check_backend(backend):
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}; got {backend!r}.")
    return backend


@dataclasses.dataclass(frozen=True)
class Settings:
    backend: str = "numba"

    def __post_init__(self):
        check_backend(self.backend)

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

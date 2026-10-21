from importlib.metadata import PackageNotFoundError, version

from . import common  # noqa: F401
from .binomial import biglasso  # noqa: F401
from .common import predict, select_model  # noqa: F401
from .viz import plot_deviance, plot_effects  # noqa: F401

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = None

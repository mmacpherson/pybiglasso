from importlib.metadata import PackageNotFoundError, version

from . import common  # noqa: F401
from .binomial import biglasso  # noqa: F401
from .path import BiglassoPath  # noqa: F401
from .viz import plot_coefficients, plot_deviance  # noqa: F401

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = None


def __getattr__(name):
    # -- the scikit-learn estimators need scikit-learn; import them on first use
    if name in ("BiglassoClassifier", "BiglassoClassifierCV"):
        try:
            from . import sklearn as _sklearn
        except ImportError as e:
            raise ImportError(
                f"{name} needs scikit-learn: pip install 'pybiglasso[sklearn]'"
            ) from e
        return getattr(_sklearn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

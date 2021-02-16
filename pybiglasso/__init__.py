# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings

import pkg_resources

from . import common  # noqa: F401
from .binomial import biglasso  # noqa: F401
from .common import predict, select_model  # noqa: F401
from .viz import plot_deviance, plot_effects  # noqa: F401

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:  # pragma: no cover
    warnings.warn(
        "Version not available for package %s (is it installed?)" % __package__, Warning
    )
    __version__ = None

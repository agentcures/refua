from importlib.metadata import PackageNotFoundError, version

from refua.api import download_assets
from refua.boltz.api import Boltz2
from refua.boltzgen.api import BoltzGen

try:  # noqa: SIM105
    __version__ = version("refua")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["Boltz2", "BoltzGen", "download_assets"]

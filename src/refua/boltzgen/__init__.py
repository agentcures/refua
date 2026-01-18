import sys
from importlib.metadata import PackageNotFoundError, version

# Keep legacy absolute imports working (e.g., `from boltzgen...`).
if "boltzgen" not in sys.modules:
    sys.modules["boltzgen"] = sys.modules[__name__]

_version_sources = ("refua", "boltzgen")
for _source in _version_sources:
    try:
        __version__ = version(_source)
        break
    except PackageNotFoundError:
        continue

import sys
from importlib.metadata import PackageNotFoundError, version

# Keep legacy absolute imports working (e.g., `from boltz...`).
if "boltz" not in sys.modules:
    sys.modules["boltz"] = sys.modules[__name__]

_version_sources = ("refua", "boltz")
for _source in _version_sources:
    try:
        __version__ = version(_source)
        break
    except PackageNotFoundError:
        continue

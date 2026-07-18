import importlib.metadata as meta

PACKAGE = "operetta_compose"

try:
    _metadata = meta.metadata(PACKAGE)
except meta.PackageNotFoundError:
    _metadata = None


def _get_urls() -> dict:
    """Parse the ``Project-URL`` metadata entries into a name -> URL mapping."""
    urls = {}
    if _metadata is not None:
        for entry in _metadata.get_all("Project-URL") or []:
            name, _, url = entry.partition(",")
            urls[name.strip()] = url.strip()
    return urls


if _metadata is not None:
    __version__ = _metadata["Version"]
    __summary__ = _metadata["Summary"]
    __authors__ = _metadata["Author-email"]
    # PEP 639 exposes the SPDX license as "License-Expression"; fall back to the
    # legacy "License" field. Use `.get()` to avoid the deprecated implicit-None
    # lookup on missing keys.
    __license__ = _metadata.get("License-Expression") or _metadata.get("License")
    __urls__ = _get_urls()
else:
    __version__ = __summary__ = __authors__ = __license__ = None
    __urls__ = {}


from . import tasks

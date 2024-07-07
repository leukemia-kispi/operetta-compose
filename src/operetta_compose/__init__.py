import importlib.metadata as meta

PACKAGE = "operetta_compose"

try:
    metadata = meta.metadata(PACKAGE)
except meta.PackageNotFoundError:
    print(f"{PACKAGE} is not installed yet.")


def _get_urls():
    __urls__ = {}
    try:
        for _url in metadata.get_all("Project-URL"):
            __urls__.update(dict([_url.replace(" ", "").split(",")]))
    except TypeError:
        __urls__["Documentation"] = None
        __urls__["Repository"] = None
        __urls__["Issues"] = None
    return __urls__


__version__ = metadata["Version"]
__summary__ = metadata["Summary"]
__authors__ = metadata["Author-email"]
__license__ = metadata["License"]
__urls__ = _get_urls()


from . import io
from . import tasks

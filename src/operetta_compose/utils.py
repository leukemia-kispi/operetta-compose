#!/usr/bin/env python3
import tqdm
import tqdm.notebook
from typing import Union
import numpy as np
import pandas as pd
import matplotlib
import string
from pathlib import Path
from pydantic import BaseModel
from typing import Optional


class OmeZarrUrl(BaseModel):
    """
    Model for a ZarrUrl
    """

    root: str
    row: Optional[str] = None
    col: Optional[str] = None
    well: Optional[str] = None
    image: Optional[str] = None


def colorbrewer(name: str):
    """Get all colors from a colormap in matplotlib

    Args:
        name: Name of the colormap
    """
    cmap = matplotlib.colormaps[name]
    colors = pd.DataFrame(
        cmap(np.linspace(0, 1, cmap.N)),
        index=range(1, cmap.N + 1),
        columns=["R", "G", "B", "A"],
    )
    return colors


def _in_notebook() -> bool:
    """Check if code is run in IPython notebook.
    The variable `__IPYTHON__` is defined in Jupyter or IPython but not a normal Python interpreter.

    Returns:
        `True` if code is executed from a Jupyter/IPython notebook
    """
    try:
        __IPYTHON__  # type: ignore
        return True
    except NameError:
        return False


def _initialize_pbar(
    n: int, msg: str
) -> Union[tqdm.std.tqdm, tqdm.notebook.tqdm_notebook]:
    """Initialize a progress bar

    Args:
        n: Number of steps
        msg: Progress bar message

    Returns:
        A tqdm progress bar
    """
    if _in_notebook():
        pbar = tqdm.notebook.tqdm(
            total=n,
            desc=msg,
            bar_format="{desc}: {n_fmt}/{total_fmt} [{remaining} s]",
        )
    else:
        pbar = tqdm.tqdm(
            total=n,
            desc=msg,
            bar_format="{desc}: {n_fmt}/{total_fmt} [{remaining} s]",
        )
    return pbar


def alpha_to_numeric(alpha: str) -> int:
    """Return the position of a single character in the alphabet

    Args:
        alpha: Single alphabet character

    Returns:
        Integer position in the alphabet
    """
    return ord(alpha.upper()) - 64


def numeric_to_alpha(numeric: int, upper: bool = True) -> str:
    """Return the upper or lowercase character for a given position in the alphabet

    Args:
        numeric: Integer position in the alphabet

    Returns:
        Single alphabet character
    """
    if upper:
        string.ascii_uppercase[numeric - 1]
    else:
        string.ascii_lowercase[numeric - 1]


def parse_zarr_url(zarr_url: str) -> OmeZarrUrl:
    """Parse the OME-ZARR URL into a dictionary with the root URL, row, column and image

    Args:
        zarr_url: Path to the OME-ZARR

    Returns:
        A `OmeZarrUrl` object
    """
    zarr_dict = {"root": None}
    if zarr_url:
        parts = [p.replace("\\", "") for p in Path(zarr_url).parts]
        for i, p in enumerate(parts):
            if p.endswith(".zarr"):
                zarr_dict["root"] = str(Path(*parts[0 : i + 1]))
                break
        if not zarr_dict["root"]:
            raise ValueError("No .zarr extension detected in URL")
        try:
            zarr_dict["row"] = parts[i + 1]
        except:
            zarr_dict["row"] = None
        try:
            zarr_dict["col"] = parts[i + 2]
            zarr_dict["well"] = zarr_dict["row"] + zarr_dict["col"]
        except:
            zarr_dict["col"] = None
        try:
            zarr_dict["image"] = parts[i + 3]
        except:
            zarr_dict["image"] = None
        return OmeZarrUrl(**zarr_dict)

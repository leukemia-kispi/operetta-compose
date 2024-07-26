import numpy as np
import pandas as pd
import logging
from pathlib import Path

import fractal_tasks_core
from pydantic import validate_call

from operetta_compose import io

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

logger = logging.getLogger(__name__)


@validate_call
def condition_registration(
    *,
    zarr_url: str,
    layout_path: str,
    condition_name: str = "condition",
    overwrite: bool = False,
) -> None:
    """Register the experimental (drug layout) in the OME-ZARR

    Args:
        zarr_url: Path to an OME-ZARR Image
        layout_path: Path to a drug layout file (.csv) with at least the columns: row, col, drug, concentration and unit
        condition_name: Name of the condition table
        overwrite: Whether to overwrite any existing OME-ZARR condition table
    """
    condition_dir = Path(f"{zarr_url}/tables/{condition_name}")
    if (not condition_dir.is_dir()) | overwrite:
        layout = pd.read_csv(
            layout_path,
            sep=None,
            engine="python",
            dtype={
                "row": str,
                "col": str,
                "drug": str,
                "concentration": np.float64,
                "unit": str,
            },
        )
        ome_zarr_url = io.parse_zarr_url(zarr_url)
        condition_table = layout.query(
            "row == @ome_zarr_url.row & col == @ome_zarr_url.col"
        )
        if not condition_table.empty:
            io.condition_to_ome_zarr(zarr_url, condition_table, condition_name)
            io.write_table_metadata(zarr_url, "condition_table", "condition")
        else:
            raise ValueError(
                "Well {row}{col} of the OME-ZARR fileset is not found in the drug layout"
            )
    else:
        raise FileExistsError(
            f"{zarr_url} already contains a condition table in the OME-ZARR fileset. To ignore the existing table set `overwrite = True`."
        )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=condition_registration,
        logger_name=logger.name,
    )

"""Register experimental conditions as a plate-level table in an OME-Zarr plate."""

import logging
from typing import Literal

import pandas as pd
from ngio import open_ome_zarr_plate
from ngio.tables import ConditionTable
from pydantic import validate_call

from operetta_compose.tasks._utils import (
    WELL_COLUMNS,
    normalize_well_columns,
    plate_url_from_zarr_url,
)
from operetta_compose.tasks._utils import (
    well_key as _well_key,
)

logger = logging.getLogger(__name__)


def read_layout(layout_path: str) -> pd.DataFrame:
    """Read an experimental layout file and normalize the well identifiers.

    The layout is a delimited text file (``.csv`` / ``.tsv``) with at least a
    ``row`` and a ``column`` (or ``col``) column identifying the well, plus an
    arbitrary number of metadata columns (e.g. drug, concentration, medium).
    This matches the condition-table format of the
    [fractal-uzh-converters](https://github.com/fractal-analytics-platform/fractal-uzh-converters)
    Operetta converter.

    Args:
        layout_path: Path to the layout file.

    Returns:
        The layout with the well identifiers renamed to the canonical ``row``
        and ``column`` columns (as strings) placed first.
    """
    layout = pd.read_csv(
        layout_path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
        quotechar='"',
    )
    row_name, column_name = normalize_well_columns(list(layout.columns))
    layout = layout.rename(columns={row_name: "row", column_name: "column"})
    for col in WELL_COLUMNS:
        # Strip stray quotes (some exports quote values) and coerce to string.
        layout[col] = (
            layout[col].astype(str).str.replace('"', "", regex=False).str.strip()
        )
    # Place the well identifiers first for readability.
    return layout[WELL_COLUMNS + [c for c in layout.columns if c not in WELL_COLUMNS]]


@validate_call
def condition_registration(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    layout_path: str,
    condition_table_name: str = "condition",
    table_backend: Literal["anndata", "json", "csv", "parquet"] = "parquet",
    overwrite: bool = False,
) -> None:
    """Register the experimental layout as a plate-level condition table.

    The layout file must contain at least the columns ``row`` and ``column``
    (or ``col``) identifying the well and can have an arbitrary number of
    additional metadata columns (e.g. drug, concentration, medium, sample).
    Only the wells that are actually present in the plate are registered.

    Args:
        zarr_urls: List of paths to the OME-Zarr images of the plate (provided
            by Fractal for a non-parallel task).
        zarr_dir: Path to the directory containing the plate (provided by
            Fractal; unused, kept for the Fractal non-parallel task signature).
        layout_path: Path to a layout file (e.g. ``.csv``) with at least the
            columns row and column (or col).
        condition_table_name: Name of the plate-level condition table to write.
        table_backend: ngio table backend used to store the condition table.
            ``parquet`` is a compact columnar default; ``csv`` keeps the table
            human-readable inside the OME-Zarr; ``anndata`` and ``json`` are
            also supported.
        overwrite: Whether to overwrite an existing condition table.
    """
    if not zarr_urls:
        raise ValueError("`zarr_urls` is empty; no plate to register conditions on.")

    plate_url = plate_url_from_zarr_url(zarr_urls[0])
    logger.info(f"Opening plate {plate_url}")
    plate = open_ome_zarr_plate(plate_url, mode="r+")

    layout = read_layout(layout_path)
    metadata_columns = [c for c in layout.columns if c not in WELL_COLUMNS]

    # Match the layout against the plate's canonical well identifiers (ngio
    # zero-pads columns, e.g. "3" -> "03") and write those canonical values so
    # the condition table lines up with the per-image tables.
    layout = layout.assign(
        _row_key=layout["row"].map(_well_key),
        _column_key=layout["column"].map(_well_key),
    )

    rows = []
    for well_path in plate.wells_paths():
        plate_row, plate_column = well_path.split("/")
        match = layout[
            (layout["_row_key"] == _well_key(plate_row))
            & (layout["_column_key"] == _well_key(plate_column))
        ]
        if match.empty:
            logger.warning(f"Well {well_path} is not present in the layout.")
            continue
        if len(match) > 1:
            logger.warning(
                f"Multiple layout entries for well {well_path}; using the first."
            )
        entry = {"row": plate_row, "column": plate_column}
        entry.update(match.iloc[0][metadata_columns].to_dict())
        rows.append(entry)

    if not rows:
        raise ValueError(
            "None of the plate wells were found in the layout file. "
            f"Plate wells: {plate.wells_paths()}."
        )
    condition_table = pd.DataFrame(rows, columns=WELL_COLUMNS + metadata_columns)

    plate.add_table(
        condition_table_name,
        ConditionTable(condition_table),
        backend=table_backend,
        overwrite=overwrite,
    )
    logger.info(
        f"Registered {len(condition_table)} condition(s) as plate-level table "
        f"`{condition_table_name}` using the `{table_backend}` backend."
    )
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=condition_registration,
        logger_name=logger.name,
    )

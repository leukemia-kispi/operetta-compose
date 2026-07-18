"""Aggregate cell counts per experimental condition across an OME-Zarr plate."""

import logging
from pathlib import Path

import pandas as pd
from ngio import open_ome_zarr_plate
from ngio.tables import GenericTable
from pydantic import validate_call

from operetta_compose.tasks._utils import (
    WELL_COLUMNS,
    plate_url_from_zarr_url,
    well_key,
)

logger = logging.getLogger(__name__)


def aggregate_cell_counts(
    feature_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    readout_column: str | None = None,
) -> pd.DataFrame:
    """Count cells per well (and readout class) and join the experimental conditions.

    The counts are computed from the feature table and joined onto the condition
    table with a left merge, so that every condition well is represented in the
    output even when no cell was measured for it (the count is 0 in that case).

    Args:
        feature_df: Concatenated feature table with at least the columns
            ``row`` and ``column`` (added by ``concatenate_image_tables``).
        condition_df: Concatenated condition table with at least the columns
            ``row`` and ``column``.
        readout_column: Optional column in the feature table used to break the
            counts down further (e.g. a classifier prediction like
            viable/dead). When ``None``, the total number of cells per well is
            counted.

    Returns:
        A long-format dataframe with one row per condition well (and readout
        class), the experimental condition columns and a ``cell_count`` column.
    """
    feature_df = feature_df.copy()
    condition_df = condition_df.copy()
    key_columns = [f"_{col}_key" for col in WELL_COLUMNS]
    for col, key in zip(WELL_COLUMNS, key_columns, strict=True):
        if col not in feature_df.columns:
            raise ValueError(
                f"Feature table is missing the `{col}` column. It should be "
                "added automatically by `concatenate_image_tables`."
            )
        if col not in condition_df.columns:
            raise ValueError(f"Condition table is missing the `{col}` column.")
        # Match on a padding-insensitive key so that feature and condition
        # tables align even if the well identifiers were stored differently
        # (e.g. "3" vs "03").
        feature_df[key] = feature_df[col].map(well_key)
        condition_df[key] = condition_df[col].map(well_key)

    group_columns = list(key_columns)
    if readout_column is not None:
        if readout_column not in feature_df.columns:
            raise ValueError(
                f"Readout column `{readout_column}` not found in the feature "
                f"table. Available columns: {sorted(feature_df.columns)}"
            )
        group_columns.append(readout_column)

    # `dropna=False` keeps readout classes that are missing/NaN rather than
    # silently discarding those cells; `observed=True` avoids materializing
    # unused combinations of categorical values.
    counts = (
        feature_df.groupby(group_columns, dropna=False, observed=True)
        .size()
        .reset_index(name="cell_count")
    )

    # One condition row per well; drop the per-image bookkeeping columns and
    # keep the condition table's (canonical) well identifiers in the output.
    conditions = condition_df.drop(
        columns=[c for c in ["path_in_well"] if c in condition_df.columns]
    ).drop_duplicates(subset=key_columns)

    # Left merge from the conditions keeps wells with zero measured cells.
    aggregated = conditions.merge(counts, on=key_columns, how="left")
    aggregated["cell_count"] = aggregated["cell_count"].fillna(0).astype(int)
    aggregated = aggregated.drop(columns=key_columns)
    return aggregated.reset_index(drop=True)


@validate_call
def cell_count_aggregation(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    feature_table_name: str = "regionprops",
    condition_table_name: str = "condition",
    readout_column: str | None = None,
    output_table_name: str = "cell_counts",
    write_csv: bool = True,
    overwrite: bool = True,
) -> None:
    """Aggregate cell counts per experimental condition across an OME-Zarr plate.

    This task reads the per-image feature and condition tables of a whole plate,
    counts the number of cells per well (optionally broken down by a readout
    column such as a classifier prediction), joins the experimental conditions
    and writes the aggregated counts back as a plate-level table (and optionally
    a CSV file).

    It is resilient to conditions for which no cell was measured: every well
    present in the condition table appears in the output with a ``cell_count``
    of 0 in that case.

    Args:
        zarr_urls: List of paths to the OME-Zarr images of the plate (provided
            by Fractal for a non-parallel task).
        zarr_dir: Path to the directory containing the plate (provided by
            Fractal). The output CSV is written here.
        feature_table_name: Name of the per-image feature table to aggregate.
        condition_table_name: Name of the plate-level condition table to join
            (written by the `condition_registration` task).
        readout_column: Optional column in the feature table to break the counts
            down by (e.g. a classifier prediction). Counts total cells per well
            when left unset.
        output_table_name: Name of the plate-level table (and CSV file) to write.
        write_csv: Whether to also export the aggregated counts as a CSV file
            next to the plate.
        overwrite: Whether to overwrite an existing output table.
    """
    if not zarr_urls:
        raise ValueError("`zarr_urls` is empty; nothing to aggregate.")

    plate_url = plate_url_from_zarr_url(zarr_urls[0])
    logger.info(f"Opening plate {plate_url}")
    plate = open_ome_zarr_plate(plate_url, mode="r+")

    # Feature tables are stored per image; `strict=False` tolerates images that
    # are missing the table (e.g. an empty well or a failed segmentation).
    feature_table = plate.concatenate_image_tables(feature_table_name, strict=False)
    # Conditions are registered as a single plate-level table (see the
    # `condition_registration` task).
    condition_table = plate.get_table(condition_table_name)

    feature_df = feature_table.dataframe.reset_index()
    condition_df = condition_table.dataframe.reset_index(drop=True)

    aggregated = aggregate_cell_counts(
        feature_df=feature_df,
        condition_df=condition_df,
        readout_column=readout_column,
    )
    logger.info(
        f"Aggregated {int(aggregated['cell_count'].sum())} cells into "
        f"{len(aggregated)} condition rows."
    )

    plate.add_table(
        output_table_name,
        GenericTable(aggregated),
        overwrite=overwrite,
    )
    logger.info(f"Wrote plate-level table `{output_table_name}`.")

    if write_csv:
        csv_path = Path(zarr_dir) / f"{output_table_name}.csv"
        aggregated.to_csv(csv_path, index=False)
        logger.info(f"Wrote CSV to {csv_path}.")

    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=cell_count_aggregation,
        logger_name=logger.name,
    )

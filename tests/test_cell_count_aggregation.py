"""Tests for the cell_count_aggregation task and its pure aggregation helper."""

import pandas as pd
from ngio import open_ome_zarr_plate
from ngio.tables import ConditionTable

from operetta_compose.tasks._utils import well_key
from operetta_compose.tasks.cell_count_aggregation import (
    aggregate_cell_counts,
    cell_count_aggregation,
)

# --- aggregate_cell_counts (pure helper) -------------------------------------


def test_aggregate_cell_counts_by_readout():
    feature_df = pd.DataFrame(
        {
            "row": ["C"] * 10,
            "column": ["03"] * 10,
            "prediction": ["viable"] * 8 + ["dead"] * 2,
        }
    )
    # C/04 has a condition but no measured cells.
    condition_df = pd.DataFrame(
        {"row": ["C", "C"], "column": ["03", "04"], "drug": ["DMSO", "Ven"]}
    )

    result = aggregate_cell_counts(
        feature_df, condition_df, readout_column="prediction"
    )

    viable = result[(result["column"] == "03") & (result["prediction"] == "viable")]
    dead = result[(result["column"] == "03") & (result["prediction"] == "dead")]
    empty_well = result[result["column"] == "04"]
    assert int(viable["cell_count"].iloc[0]) == 8
    assert int(dead["cell_count"].iloc[0]) == 2
    # Resilience: the empty condition still appears, with a count of 0 for
    # every observed readout category (viable and dead).
    assert int(empty_well["cell_count"].sum()) == 0
    assert set(empty_well["drug"]) == {"Ven"}
    assert set(empty_well["prediction"]) == {"viable", "dead"}


def test_aggregate_cell_counts_total_and_padding():
    # Feature identifiers ("3") differ in padding from conditions ("03").
    feature_df = pd.DataFrame({"row": ["C", "C", "C"], "column": ["3", "3", "3"]})
    condition_df = pd.DataFrame(
        {"row": ["C", "C"], "column": ["03", "04"], "drug": ["DMSO", "Ven"]}
    )

    result = aggregate_cell_counts(feature_df, condition_df, readout_column=None)

    counts = dict(zip(result["column"], result["cell_count"]))
    assert counts["03"] == 3  # padding-insensitive match
    assert counts["04"] == 0


# --- cell_count_aggregation (task) -------------------------------------------


def test_cell_count_aggregation(
    plate_url, tmp_path, build_plate, add_feature_table, image_urls, wells
):
    plate = build_plate(plate_url, wells)
    # C/3 has measured cells; C/4 has a condition but no feature table.
    add_feature_table(plate, "C", "3", ["viable"] * 8 + ["dead"] * 2)
    condition_df = pd.DataFrame(
        {
            "row": ["C", "C"],
            "column": ["03", "04"],
            "drug": ["DMSO", "Venetoclax"],
            "concentration": [0.0, 1.0],
        }
    )
    plate.add_table("condition", ConditionTable(condition_df), overwrite=True)

    cell_count_aggregation(
        zarr_urls=image_urls(plate_url, wells),
        zarr_dir=str(tmp_path),
        feature_table_name="regionprops",
        condition_table_name="condition",
        readout_column="prediction",
        output_table_name="cell_counts",
        table_backend="csv",
        overwrite=True,
    )

    result = open_ome_zarr_plate(plate_url).get_table("cell_counts").dataframe
    result["cell_count"] = result["cell_count"].astype(int)
    # The csv backend round-trips `column` without ngio's zero-padding
    # ("03" -> 3), so match on the padding-insensitive well key instead.
    result["_col"] = result["column"].map(well_key)

    def count(column, prediction):
        sel = result[result["_col"] == well_key(column)]
        if prediction is not None:
            sel = sel[sel["prediction"] == prediction]
        return int(sel["cell_count"].sum())

    assert count("03", "viable") == 8
    assert count("03", "dead") == 2
    # Well C/4 has a condition but no cells -> a resilient 0 for each category.
    assert count("04", "viable") == 0
    assert count("04", "dead") == 0
    assert "Venetoclax" in set(result["drug"])
    assert int(result["cell_count"].sum()) == 10

"""Tests for the operetta-compose plate-level tasks.

The tests build a small synthetic OME-Zarr plate with ngio so that they are
self-contained and do not depend on real microscopy data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ngio import (
    ImageInWellPath,
    create_empty_plate,
    create_ome_zarr_from_array,
    open_ome_zarr_plate,
)
from ngio.tables import ConditionTable, FeatureTable

from operetta_compose.tasks._utils import well_key
from operetta_compose.tasks.cell_count_aggregation import (
    aggregate_cell_counts,
    cell_count_aggregation,
)
from operetta_compose.tasks.condition_registration import condition_registration
from operetta_compose.tasks.spot_detection import count_spots_per_label

TEST_DIR = Path(__file__).resolve().parent
FIXTURES = TEST_DIR / "fixtures"

# "C" / "3" matches tests/fixtures/drug_layout.csv (ngio stores it as "C/03").
WELLS = [("C", "3"), ("C", "4")]


def _build_plate(plate_url, wells):
    """Create an OME-Zarr plate with one (blank) image per well."""
    images = [ImageInWellPath(row=r, column=c, path="0") for r, c in wells]
    plate = create_empty_plate(
        plate_url, name="test_plate", images=images, overwrite=True
    )
    for r, c in wells:
        store = plate.get_image_store(row=r, column=c, image_path="0")
        create_ome_zarr_from_array(
            store=store,
            array=np.zeros((1, 16, 16), dtype=np.uint16),
            pixelsize=0.5,
            levels=1,
            overwrite=True,
        )
    return plate


def _add_feature_table(plate, row, column, readouts, name="regionprops"):
    """Add a per-image feature table with a `prediction` readout column."""
    container = plate.get_image(row=row, column=column, image_path="0")
    n = len(readouts)
    df = pd.DataFrame(
        {
            "label": np.arange(1, n + 1),
            "area": np.linspace(1.0, 2.0, n),
            "prediction": readouts,
        }
    )
    container.add_table(name, FeatureTable(df), overwrite=True)


def _image_urls(plate_url, wells):
    return [str(Path(plate_url) / r / well_key(c).zfill(2) / "0") for r, c in wells]


@pytest.fixture
def plate_url(tmp_path):
    return str(tmp_path / "operetta_plate.zarr")


# --- condition_registration --------------------------------------------------


def test_condition_registration(plate_url, tmp_path):
    _build_plate(plate_url, WELLS)
    condition_registration(
        zarr_urls=_image_urls(plate_url, WELLS),
        zarr_dir=str(tmp_path),
        layout_path=str(FIXTURES / "drug_layout.csv"),
        condition_table_name="condition",
        overwrite=True,
    )

    plate = open_ome_zarr_plate(plate_url)
    assert "condition" in plate.list_tables()
    condition = plate.get_table("condition").dataframe

    # The layout only defines well C/3, which ngio stores as column "03".
    assert list(condition["row"]) == ["C"]
    assert list(condition["column"]) == ["03"]
    row = condition.iloc[0]
    assert row["drug"] == "DMSO"
    assert float(row["concentration"]) == pytest.approx(0.125)


def test_condition_registration_no_matching_well(plate_url, tmp_path):
    # Plate wells that are absent from the layout must raise.
    _build_plate(plate_url, [("A", "1"), ("A", "2")])
    with pytest.raises(ValueError, match="None of the plate wells"):
        condition_registration(
            zarr_urls=_image_urls(plate_url, [("A", "1")]),
            zarr_dir=str(tmp_path),
            layout_path=str(FIXTURES / "drug_layout.csv"),
            overwrite=True,
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
    # Resilience: the empty condition still appears, with a count of 0.
    assert int(empty_well["cell_count"].sum()) == 0
    assert list(empty_well["drug"]) == ["Ven"]


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


def test_cell_count_aggregation(plate_url, tmp_path):
    plate = _build_plate(plate_url, WELLS)
    # C/3 has measured cells; C/4 has a condition but no feature table.
    _add_feature_table(plate, "C", "3", ["viable"] * 8 + ["dead"] * 2)
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
        zarr_urls=_image_urls(plate_url, WELLS),
        zarr_dir=str(tmp_path),
        feature_table_name="regionprops",
        condition_table_name="condition",
        readout_column="prediction",
        output_table_name="cell_counts",
        write_csv=True,
        overwrite=True,
    )

    result = open_ome_zarr_plate(plate_url).get_table("cell_counts").dataframe
    result["cell_count"] = result["cell_count"].astype(int)

    def count(column, prediction):
        sel = result[result["column"] == column]
        if prediction is not None:
            sel = sel[sel["prediction"] == prediction]
        return int(sel["cell_count"].sum())

    assert count("03", "viable") == 8
    assert count("03", "dead") == 2
    # Well C/4 has a condition but no cells -> resilient zero count.
    assert count("04", None) == 0
    assert "Venetoclax" in set(result["drug"])

    csv_path = tmp_path / "cell_counts.csv"
    assert csv_path.exists()
    assert int(pd.read_csv(csv_path)["cell_count"].sum()) == 10


# --- count_spots_per_label (pure helper) -------------------------------------


def test_count_spots_per_label():
    # Two labelled regions in a 10x10 image.
    label = np.zeros((10, 10), dtype=int)
    label[0:5, 0:5] = 1
    label[5:10, 5:10] = 2
    # 3 spots on label 1, 1 spot on label 2, 1 spot on the background.
    points = np.array([[1, 1], [2, 2], [3, 3], [7, 7], [0, 9]])

    df = count_spots_per_label(points, label)

    assert df.index.name == "label"
    counts = df["spot_count"].to_dict()
    assert counts[1] == 3
    assert counts[2] == 1
    # The background spot (0, 9) is ignored; only the two labels are reported.
    assert set(counts) == {1, 2}


def test_count_spots_per_label_no_spots():
    # A label with no detected spots must still appear with a count of 0.
    label = np.zeros((6, 6), dtype=int)
    label[0:3, 0:3] = 5
    df = count_spots_per_label(np.empty((0, 2)), label)
    assert df.index.tolist() == [5]
    assert df["spot_count"].tolist() == [0]


def test_count_spots_per_label_clips_out_of_bounds():
    # Subpixel detections just outside the image are clipped back inside.
    label = np.zeros((4, 4), dtype=int)
    label[3, 3] = 7
    points = np.array([[3.9, 3.9]])  # floors to (3, 3) after clipping
    df = count_spots_per_label(points, label)
    assert df.loc[7, "spot_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

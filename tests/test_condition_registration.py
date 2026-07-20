"""Tests for the condition_registration task."""

import pytest
from ngio import open_ome_zarr_plate

from operetta_compose.tasks.condition_registration import condition_registration


def test_condition_registration(
    plate_url, tmp_path, build_plate, image_urls, wells, fixtures_dir
):
    build_plate(plate_url, wells)
    condition_registration(
        zarr_urls=image_urls(plate_url, wells),
        zarr_dir=str(tmp_path),
        layout_path=str(fixtures_dir / "drug_layout.csv"),
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


def test_condition_registration_no_matching_well(
    plate_url, tmp_path, build_plate, image_urls, fixtures_dir
):
    # Plate wells that are absent from the layout must raise.
    build_plate(plate_url, [("A", "1"), ("A", "2")])
    with pytest.raises(ValueError, match="None of the plate wells"):
        condition_registration(
            zarr_urls=image_urls(plate_url, [("A", "1")]),
            zarr_dir=str(tmp_path),
            layout_path=str(fixtures_dir / "drug_layout.csv"),
            overwrite=True,
        )

"""Shared fixtures for the operetta-compose task tests.

The plate builders create small synthetic OME-Zarr plates with ngio so that the
tests are self-contained and do not depend on real microscopy data. Helpers are
exposed as fixtures returning callables so test modules can request them without
importing from ``conftest``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ngio import (
    ImageInWellPath,
    create_empty_plate,
    create_ome_zarr_from_array,
)
from ngio.tables import FeatureTable

from operetta_compose.tasks._utils import well_key

TEST_DIR = Path(__file__).resolve().parent


@pytest.fixture
def fixtures_dir():
    """Directory holding the static test fixtures (e.g. drug_layout.csv)."""
    return TEST_DIR / "fixtures"


@pytest.fixture
def wells():
    """Default wells. "C"/"3" matches fixtures/drug_layout.csv ("C/03" in ngio)."""
    return [("C", "3"), ("C", "4")]


@pytest.fixture
def plate_url(tmp_path):
    return str(tmp_path / "operetta_plate.zarr")


@pytest.fixture
def build_plate():
    """Return a callable that creates a plate with one blank image per well."""

    def _build_plate(plate_url, wells):
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

    return _build_plate


@pytest.fixture
def add_feature_table():
    """Return a callable adding a per-image feature table with a readout column."""

    def _add_feature_table(plate, row, column, readouts, name="regionprops"):
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

    return _add_feature_table


@pytest.fixture
def image_urls():
    """Return a callable building the OME-Zarr image URLs for the given wells."""

    def _image_urls(plate_url, wells):
        return [str(Path(plate_url) / r / well_key(c).zfill(2) / "0") for r, c in wells]

    return _image_urls

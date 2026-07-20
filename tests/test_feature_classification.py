"""Tests for the feature_classification task.

The task runs a trained scikit-learn classifier loaded from a neutral joblib
bundle (tests/fixtures/classifier.joblib), so it needs no napari-feature-
classifier / napari and runs in the default environment.
"""

import ngio
import numpy as np
import pandas as pd
import pytest
from ngio.tables import FeatureTable

from operetta_compose.tasks.feature_classification import feature_classification

# Feature columns the bundled classifier was trained on, and the class names it
# predicts (see tests/fixtures/classifier.joblib).
CLASSIFIER_FEATURES = [
    "area",
    "area_convex",
    "intensity_mean",
    "intensity_max",
    "intensity_min",
    "intensity_std",
    "eccentricity",
    "perimeter",
]
CLASS_NAMES = {"MSC", "Viable leukemia", "Dead cells"}


def test_feature_classification(plate_url, build_plate, image_urls, fixtures_dir):
    plate = build_plate(plate_url, [("C", "3")])

    # A regionprops table carrying exactly the features the classifier expects.
    n = 6
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f: rng.uniform(0.1, 100.0, n) for f in CLASSIFIER_FEATURES})
    df["label"] = np.arange(1, n + 1)
    plate.get_image(row="C", column="3", image_path="0").add_table(
        "regionprops", FeatureTable(df), overwrite=True
    )

    image_url = image_urls(plate_url, [("C", "3")])[0]
    feature_classification(
        zarr_url=image_url,
        classifier_path=str(fixtures_dir / "classifier.joblib"),
    )

    result = ngio.open_ome_zarr_container(image_url).get_table("regionprops").dataframe
    # The prediction column is named after the classifier file ("classifier").
    assert "classifier_prediction" in result.columns
    assert len(result) == n

    predictions = set(result["classifier_prediction"])
    assert predictions <= CLASS_NAMES  # only valid class names, all rows classified


def test_feature_classification_accepts_clf(
    plate_url, build_plate, image_urls, fixtures_dir
):
    # A real `.clf` exported from the napari-feature-classifier plugin
    # (tests/fixtures/classifier.clf) is a full pickled Classifier and must
    # load and classify headlessly through feature-classifier-core, without
    # napari-feature-classifier installed.
    plate = build_plate(plate_url, [("C", "3")])
    n = 6
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f: rng.uniform(0.1, 100.0, n) for f in CLASSIFIER_FEATURES})
    df["label"] = np.arange(1, n + 1)
    plate.get_image(row="C", column="3", image_path="0").add_table(
        "regionprops", FeatureTable(df), overwrite=True
    )

    image_url = image_urls(plate_url, [("C", "3")])[0]
    feature_classification(
        zarr_url=image_url,
        classifier_path=str(fixtures_dir / "classifier.clf"),
    )

    result = ngio.open_ome_zarr_container(image_url).get_table("regionprops").dataframe
    assert "classifier_prediction" in result.columns
    assert len(result) == n
    assert set(result["classifier_prediction"]) <= CLASS_NAMES


def test_feature_classification_missing_feature_raises(
    plate_url, build_plate, image_urls, fixtures_dir
):
    plate = build_plate(plate_url, [("C", "3")])
    # A feature table missing the columns the classifier needs.
    df = pd.DataFrame({"label": [1, 2, 3], "area": [1.0, 2.0, 3.0]})
    plate.get_image(row="C", column="3", image_path="0").add_table(
        "regionprops", FeatureTable(df), overwrite=True
    )

    with pytest.raises(ValueError, match="missing columns required by the classifier"):
        feature_classification(
            zarr_url=image_urls(plate_url, [("C", "3")])[0],
            classifier_path=str(fixtures_dir / "classifier.joblib"),
        )

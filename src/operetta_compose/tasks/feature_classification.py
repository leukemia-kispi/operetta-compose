import logging
import re

import ngio
import ngio.tables
import pandas as pd
from feature_classifier_core import Classifier
from pydantic import validate_call

logger = logging.getLogger(__name__)


@validate_call
def feature_classification(
    *,
    zarr_url: str,
    classifier_path: str,
    table_name: str = "regionprops",
    classifier_name: str | None = None,
) -> None:
    """Classify cells with a trained classifier and write them to the OME-Zarr.

    The classifier can be a full `.clf` file or a neutral joblib bundle
    with following three keys
     - ``estimator``: the fitted scikit-learn model
     - ``feature_names``: the feature columns
     - ``class_names``: the human-readable label per class

    Args:
        zarr_url: Path to an OME-ZARR Image
        classifier_path: Path to the joblib classifier bundle
        table_name: Folder name of the measured regionprops features
        classifier_name: Name of the classification results to be written to
            the feature table. It will default to the name of the classifier +
            "_prediction" when left unset.
    """
    if classifier_name is None:
        classifier_filename = classifier_path.split("/")[-1].split(".")[0]
        classifier_name = re.sub(r"[\W]+", "_", classifier_filename) + "_prediction"

    # feature-classifier-core loads either a neutral bundle or a full `.clf`
    # and validates the bundle format, so that logic lives in one place.
    clf = Classifier.load(classifier_path)
    estimator = clf.get_estimator()
    feature_names = clf.get_feature_names()
    class_names = clf.get_class_names()

    ome_zarr_container = ngio.open_ome_zarr_container(zarr_url)
    feature_table = ome_zarr_container.get_table(
        name=table_name, check_type="feature_table"
    )
    features = feature_table.dataframe.reset_index()
    if "label" not in features.columns:
        raise ValueError(
            "The feature table does not contain a label column. "
            "Please check the table name and the feature table."
        )

    missing = [f for f in feature_names if f not in features.columns]
    if missing:
        raise ValueError(
            f"The feature table is missing columns required by the classifier: "
            f"{missing}. Available columns: {sorted(features.columns)}"
        )

    # Rows with missing feature values cannot be classified; label them "NaN"
    # rather than dropping them (matches the napari-feature-classifier behavior).
    feature_matrix = features[feature_names]
    valid = feature_matrix.notna().all(axis=1)
    predictions = pd.Series("NaN", index=features.index, dtype=object)
    if valid.any():
        classes = estimator.predict(feature_matrix[valid])
        predictions.loc[valid] = [class_names[int(c) - 1] for c in classes]
    features[classifier_name] = predictions.to_numpy()

    new_feature_table = ngio.tables.FeatureTable(
        features,
        reference_label=feature_table.reference_label,
    )
    # Write the table to disk again
    ome_zarr_container.add_table(
        name=table_name, table=new_feature_table, overwrite=True
    )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=feature_classification,
        logger_name=logger.name,
    )

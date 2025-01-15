import pandas as pd
import logging
import ngio

import fractal_tasks_core
from pydantic import validate_call



__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

logger = logging.getLogger(__name__)


@validate_call
def feature_classification(
    *,
    zarr_url: str,
    classifier_path: str,
    table_name: str = "regionprops",
) -> None:
    """Classify cells using the [napari-feature-classifier](https://github.com/fractal-napari-plugins-collection/napari-feature-classifier) and write them to the OME-ZARR

    Args:
        zarr_url: Path to an OME-ZARR Image
        classifier_path: Path to the pickled scikit-learn classifier
        table_name: Folder name of the measured regionprobs features
    """
    with open(classifier_path, "rb") as f:
        clf = pd.read_pickle(f)

    zarr_img = ngio.NgffImage(zarr_url)
    feature_table = zarr_img.tables.get_table(name=table_name)

    features = feature_table.table.reset_index()
    if "prediction" in features.columns:
        features = features.drop(columns=["prediction"])

    remove_roi_id_column = False
    index_columns = ["roi_id", "label"]
    if "roi_id" not in features.columns:
        features["roi_id"] = zarr_url
        remove_roi_id_column = True

    # Select feature subset in expected order
    features = features[clf.get_feature_names() + index_columns]

    # Run predictions
    predictions = clf.predict(features).reset_index()

    # Fuse into existing feature table
    features_with_predictions = features.merge(
        predictions,
        on=index_columns,
        how="outer"
    )
    if remove_roi_id_column:
        features_with_predictions = features_with_predictions.drop(columns="roi_id")

    # Write the table to disk again
    new_table = zarr_img.tables.new(
        name=table_name,
        table_type="feature_table",
        label_image=f"../{feature_table.source_label()}",
        overwrite=True
    )
    new_table.set_table(features_with_predictions)
    new_table.consolidate()


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=feature_classification,
        logger_name=logger.name,
    )

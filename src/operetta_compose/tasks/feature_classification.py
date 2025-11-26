import ngio.tables
import pandas as pd
import logging
import ngio
import re
from typing import Optional

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
    classifier_name: Optional[str] = None,
) -> None:
    """Classify cells using the [napari-feature-classifier](https://github.com/fractal-napari-plugins-collection/napari-feature-classifier) and write them to the OME-ZARR

    Args:
        zarr_url: Path to an OME-ZARR Image
        classifier_path: Path to the pickled scikit-learn classifier
        table_name: Folder name of the measured regionprobs features
        classifier_name: Name of the classification results to be written to
            the feature table. It will default to the name of the classifier +
            "_prediction" when left unset.
    """
    if classifier_name is None:
        classifier_filename = classifier_path.split("/")[-1].split(".")[0]
        classifier_name = re.sub(r'[\W]+', '_', classifier_filename) + "_prediction"

    with open(classifier_path, "rb") as f:
        try:
            clf = pd.read_pickle(f)
        except AttributeError as e:
            raise AttributeError(
                "Loading the classifier failed. The most likely reason is: "
                "The classifier was trained with a different classifier "
                "plugin version (likely napari-feature-classifier <= 0.2.1)."
                "This version of the operetta-compose task is not compatible "
                "with that classifier version. Use an older version like "
                "operetta-compose 0.2.12."
                f"Original error: {e}"
            )

    ome_zarr_container = ngio.open_ome_zarr_container(zarr_url)
    feature_table = ome_zarr_container.get_table(name=table_name, check_type="feature_table")
    features = feature_table.dataframe
    features = features.reset_index()
    if "label" not in features.columns:
        raise ValueError(
            "The feature table does not contain a label column. "
            "Please check the table name and the feature table."
        )

    if classifier_name in features.columns:
        features = features.drop(columns=[classifier_name])

    remove_roi_id_column = False
    index_columns = ["roi_id", "label"]
    if "roi_id" not in features.columns:
        features["roi_id"] = zarr_url
        remove_roi_id_column = True

    # Select feature subset in expected order
    features_subset = features[clf.get_feature_names() + index_columns]

    # Run predictions & save name of prediction in dataframe
    predictions = clf.predict(features_subset).reset_index()
    predictions[classifier_name] = predictions['prediction'].map(
        lambda x: clf._class_names[int(x) - 1] if pd.notna(x) else "NaN"
    )
    predictions = predictions.drop(columns="prediction")

    # Fuse into existing feature table
    features_with_predictions = features.merge(
        predictions,
        on=index_columns,
        how="outer"
    )
    if remove_roi_id_column:
        features_with_predictions = features_with_predictions.drop(columns="roi_id")

    new_feature_table = ngio.tables.FeatureTable(
        dataframe=features_with_predictions,
        reference_label=feature_table.reference_label
    )
    # Write the table to disk again
    ome_zarr_container.add_table(
        name=table_name,
        table = new_feature_table,
        overwrite=True
    )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=feature_classification,
        logger_name=logger.name,
    )

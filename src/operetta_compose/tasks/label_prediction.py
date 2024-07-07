import numpy as np
import pandas as pd
import anndata as ad
import logging
import pickle
from zarr.errors import PathNotFoundError

import fractal_tasks_core
from pydantic.v1.decorator import validate_arguments

from operetta_compose import io
from operetta_compose import utils

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

logger = logging.getLogger(__name__)


def label_prediction(
    *,
    zarr_url: str,
    classifier_path: str,
    feature_name: str = "regionprops",
) -> None:
    """Make predictions on the selected wells and write them to the OME-ZARR

    Args:
        zarr_url: Path to an OME-ZARR Image
        classifier_path: Path to the pickled scikit-learn classifier
        feature_name: Folder name of the measured regionprobs features
    """
    with open(classifier_path, "rb") as f:
        clf = pickle.loads(f.read())

    try:
        ann_tbl = ad.read_zarr(f"{zarr_url}/tables/{feature_name}")
    except PathNotFoundError:
        raise FileNotFoundError(
            f"No measurements exist at the specified zarr URL in the table {feature_name}."
        )
    ann_tbl.obs = ann_tbl.obs.drop(["roi_id", "prediction"], axis=1, errors="ignore")

    ome_zarr_url = utils.parse_zarr_url(zarr_url)
    ann_tbl.obs["roi_id"] = ome_zarr_url.well
    data = pd.concat((ann_tbl.obs, ann_tbl.to_df()), axis=1)
    ann_tbl.obs = ann_tbl.obs.merge(clf.predict(data).reset_index())
    ann_tbl.obs_names = ann_tbl.obs.index.map(str)
    ann_tbl.write_zarr(f"{zarr_url}/tables/{feature_name}")


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=label_prediction,
        logger_name=logger.name,
    )

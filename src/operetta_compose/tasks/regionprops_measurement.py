import numpy as np
import pandas as pd
from pathlib import Path
import logging
from skimage.measure import regionprops_table

import fractal_tasks_core
from pydantic.decorator import validate_arguments

from operetta_compose import io

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

logger = logging.getLogger(__name__)

PROPS = [
    "area",
    "bbox",
    "area_bbox",
    "moments_central",
    "centroid",
    "area_convex",
    "image_convex",
    "coords",
    "eccentricity",
    "equivalent_diameter_area",
    "euler_number",
    "extent",
    "feret_diameter_max",
    "area_filled",
    "image_filled",
    "moments_hu",
    "image",
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "image_intensity",
    "centroid_local",
    "axis_major_length",
    "intensity_max",
    "intensity_mean",
    "intensity_min",
    "axis_minor_length",
    "moments",
    "moments_normalized",
    "orientation",
    "perimeter",
    "perimeter_crofton",
    "slice",
    "solidity",
    "moments_weighted_central",
    "moments_weighted_central",
    "centroid_weighted",
    "moments_weighted_hu",
    "centroid_weighted_local",
    "moments_weighted",
    "moments_weighted_normalized",
    "label",
]


@validate_arguments
def regionprops_measurement(
    *, zarr_url: str, feature_name: str = "regionprops", level: int = 0, overwrite=False
) -> None:
    """Take measurements using regionprobs and write the features to the OME-ZARR

    Args:
        zarr_url: Path to an OME-ZARR Image
        feature_name: Folder name of the measured regionprobs features
        level: Resolution level (0 = full resolution)
        overwrite: Whether to overwrite any existing OME-ZARR feature table
    """
    feature_dir = Path(f"{zarr_url}/tables/{feature_name}")
    if (not feature_dir.is_dir()) | overwrite:
        roi_url, roi_idx = io.get_roi(zarr_url, "well_ROI_table", level)
        img = io.load_intensity_roi(roi_url, roi_idx)
        labels = io.load_label_roi(roi_url, roi_idx)
        tbl = feature_table(labels[0], img[0])  # FIXME generalize for 3D
        io.features_to_ome_zarr(zarr_url, tbl, feature_name)
    else:
        raise FileExistsError(
            f"{zarr_url} already contains a feature table in the OME-ZARR. To ignore the existing table set `overwrite = True`."
        )


def feature_table(
    labels: np.ndarray,
    img: np.ndarray,
    properties: list[str] = [
        "label",
        "area",
        "intensity_mean",
        "intensity_max",
        "intensity_min",
        "eccentricity",
        "perimeter",
        "centroid",
        "solidity",
    ],
) -> pd.DataFrame:
    """Generate a regionprobs feature table

    Args:
        labels: A labels array
        img: An intensity array
        properties: A list of regionprobs properties

    Returns:
        A feature dataframe including a column with the label index
    """
    props = regionprops_table(labels, img, properties=properties)
    features = pd.DataFrame(props)
    label_col = features.pop("label")
    features.insert(0, "label", label_col)
    return features


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=regionprops_measurement,
        logger_name=logger.name,
    )

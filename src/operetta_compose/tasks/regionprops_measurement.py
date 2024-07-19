import numpy as np
import pandas as pd
from pathlib import Path
import logging
from skimage.measure import regionprops_table

import fractal_tasks_core
from pydantic.v1.decorator import validate_arguments

from operetta_compose import io

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

logger = logging.getLogger(__name__)

PROPS = [
    "label",
    "area",
    "bbox",
    "area_bbox",
    "moments_central",
    "area_convex",
    "equivalent_diameter_area",
    "euler_number",
    "extent",
    "feret_diameter_max",
    "area_filled",
    "moments_hu",
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "centroid_local",
    "axis_major_length",
    "intensity_mean",
    "intensity_max",
    "intensity_min",
    "eccentricity",
    "axis_minor_length",
    "moments",
    "moments_normalized",
    "orientation",
    "perimeter",
    "perimeter_crofton",
    "centroid",
    "solidity",
    "moments_weighted_central",
    "moments_weighted_central",
    "centroid_weighted",
    "moments_weighted_hu",
    "centroid_weighted_local",
    "moments_weighted",
    "moments_weighted_normalized",
]


@validate_arguments
def regionprops_measurement(
    *,
    zarr_url: str,
    table_name: str = "regionprops",
    label_name: str = "nuclei",
    level: int = 0,
    overwrite=False,
) -> None:
    """Take measurements using regionprobs and write the features to the OME-ZARR

    Args:
        zarr_url: Path to an OME-ZARR Image
        table_name: Folder name of the measured regionprobs features
        label_name: Name of the labels to use for feature measurements
        level: Resolution level (0 = full resolution)
        overwrite: Whether to overwrite any existing OME-ZARR feature table
    """
    feature_dir = Path(f"{zarr_url}/tables/{table_name}")
    if (not feature_dir.is_dir()) | overwrite:
        roi_url, roi_idx = io.get_roi(zarr_url, "well_ROI_table", level)
        img = io.load_intensity_roi(roi_url, roi_idx)
        labels = io.load_label_roi(roi_url, roi_idx, name=label_name)
        tbl = feature_table(labels[0], img[0], PROPS)  # FIXME generalize for 3D
        io.features_to_ome_zarr(zarr_url, tbl, table_name, label_name)
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
        properties: A list of regionprops properties

    Returns:
        A feature dataframe including a column with the label index
    """
    props = regionprops_table(labels, img, properties=properties)
    features = pd.DataFrame(props)
    features.insert(0, "label", features.pop("label"))
    return features


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=regionprops_measurement,
        logger_name=logger.name,
    )

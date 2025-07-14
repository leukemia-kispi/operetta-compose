import numpy as np
import pandas as pd
from pathlib import Path
import logging
from skimage.measure import regionprops_table

import fractal_tasks_core
from pydantic import validate_call

from operetta_compose import io

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

logger = logging.getLogger(__name__)

# Documentation of features at https://scikit-image.org/docs/stable/api/skimage.measure.html
PROPS = [
    "label",
    "area",
    "area_convex",
    "intensity_mean",
    "intensity_max",
    "intensity_min",
    "intensity_std",
    "eccentricity",
    "perimeter",
    "perimeter_crofton",
    "solidity",
    "equivalent_diameter_area",
    "feret_diameter_max",
    "axis_major_length",
    "axis_minor_length",
    "orientation",
    "extent",
    # "inertia_tensor",
    "inertia_tensor_eigvals",
    "bbox",
    "area_bbox",
    "centroid",
    "centroid_weighted",
    # "centroid_local",
    # "centroid_weighted_local",
    # "moments",
    # "moments_normalized",
    # "moments_weighted",
    # "moments_central",
    # "moments_hu",
    # "moments_weighted_normalized",
    # "moments_weighted_central",
    # "moments_weighted_hu",
]

only_2d = [
    "eccentricity",
    "perimeter",
    "perimeter_crofton",
    "orientation",
    "axis_major_length",
    "axis_minor_length",
]

# define which props are purely intensity‐based (others could also be added which use the image_intensity table)
INTENSITY_PROPS = [
    "intensity_mean",
    "intensity_max",
    "intensity_min",
    "intensity_std",
]



@validate_call
def regionprops_measurement(
    *,
    zarr_url: str,
    table_name: str = "regionprops",
    label_name: str = "nuclei",
    level: int = 0,
    overwrite: bool = False,
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

        # drop the singleton Z slice (edit here if we have multiple z-slices)
        if img.ndim == 4 and img.shape[1] == 1:
            img = img[:, 0, ...]        # → (C, Y, X)
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0, ...]  # → (1, Y, X)

        #  move channel axis to last on both
        #    img:    (C, Y, X) → (Y, X, C)
        #    labels: (1, Y, X) → (Y, X, 1)
        img    = np.moveaxis(img, 0, -1)
        labels = np.moveaxis(labels, 0, -1)
        base_props = [p for p in PROPS if p not in INTENSITY_PROPS]

        #for multi‑channel images
        if img.ndim == 3 and img.shape[-1] > 1:
            # always strip out that singleton label channel when calling feature_table
            label2d = labels[..., 0]   # shape (Y, X)

            # channel‑0: do morphology + intensity
            tbl = feature_table(
                label2d,
                img[..., 0],
                base_props + INTENSITY_PROPS
            )
            tbl = tbl.rename({p: f"{p}_ch0" for p in INTENSITY_PROPS}, axis=1)

            # channels 1…C–1: only intensity
            for c in range(1, img.shape[-1]):
                # include 'label' so we can merge
                props = ["label"] + INTENSITY_PROPS
                ch_tbl = feature_table(label2d, img[..., c], props)
                ch_tbl = ch_tbl.rename(
                    {p: f"{p}_ch{c}" for p in INTENSITY_PROPS},
                    axis=1
                )
                tbl = tbl.merge(ch_tbl, on="label", how="outer")
        # for single channel images    
        else:
            if img.shape[-1] == 1:
                img = img[0]
                labels = labels[0]
                properties = PROPS
            else:
                properties = [p for p in PROPS if p not in only_2d]
            tbl = feature_table(labels, img, properties)
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
    if labels.ndim == img.ndim + 1 and labels.shape[-1] == 1:
       labels = labels[..., 0]
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

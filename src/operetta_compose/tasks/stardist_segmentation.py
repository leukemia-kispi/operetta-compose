import numpy as np
import logging
from pathlib import Path
import shutil
from zarr.errors import ContainsArrayError
from csbdeep.utils import normalize
from stardist.models import StarDist2D, StarDist3D

from pydantic.v1.decorator import validate_arguments

import fractal_tasks_core
from fractal_tasks_core.channels import ChannelInputModel, ChannelNotFoundError
from fractal_tasks_core.channels import get_channel_from_image_zarr

from operetta_compose import io

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

logger = logging.getLogger(__name__)


@validate_arguments
def stardist_segmentation(
    *,
    zarr_url: str,
    channel: ChannelInputModel,
    roi_table: str = "FOV_ROI_table",
    stardist_model: str = "2D_versatile_fluo",
    label_name: str = "nuclei",
    prob_thresh: float = None,
    nms_thresh: float = None,
    scale: int = 1,
    level: int = 0,
    overwrite: bool = False,
) -> None:
    """Segment cells with Stardist

    Args:
        zarr_url: Path to an OME-ZARR Image
        roi_table: Name of the ROI table
        roi:  Index of the ROI in the selected well
        channel: Channel for segmentation; requires either `wavelength_id` (e.g. `A01_C01`) or `label` (e.g. `DAPI`) but not both
        stardist_model: Name of the Stardist model ("2D_versatile_fluo", "2D_versatile_he", "2D_demo", "3D_demo")
        label_name: Name of the labels folder
        prob_thresh: prob_thresh: Only consider objects with predicted object probability above this threshold
        nms_thresh: Perform non-maximum suppression (NMS) that considers two objects to be the same when their area/surface overlap exceeds this threshold
        scale: Scale the input image internally by a factor and rescale the output accordingly.
        level: Resolution level (0 = full resolution)
        overwrite: Whether to overwrite any existing OME-ZARR segmentations
    """
    if "3D" in stardist_model:
        model = StarDist3D.from_pretrained(stardist_model)
    else:
        model = StarDist2D.from_pretrained(stardist_model)

    roi = 0
    curr_roi_max = 0
    roi_url, roi_idx = io.get_roi(zarr_url, roi_table, level)

    try:
        channel_idx = get_channel_from_image_zarr(
            image_zarr_path=zarr_url,
            wavelength_id=channel.wavelength_id,
            label=channel.label,
        ).index
    except ChannelNotFoundError as e:
        logger.warning(
            f"Channel with wavelength_id: {channel.wavelength_id} "
            f"and label: {channel.label} not found, exit from the task.\n"
            f"Original error: {str(e)}"
        )
        return None

    labels = np.empty(
        (
            1,
            roi_idx["e_z"].max(),
            roi_idx["e_y"].max(),
            roi_idx["e_x"].max(),
        ),
        dtype=np.uint16,
    )
    while True:
        try:
            img = io.load_intensity_roi(roi_url, roi_idx, roi, channel_idx)
        except KeyError:
            break
        if not "3D" in stardist_model:
            img = img[0]
        roi_labels, _ = model.predict_instances(
            normalize(img),
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            scale=scale,
        )
        roi_max = roi_labels.max()
        roi_labels[roi_labels != 0] += curr_roi_max

        labels[
            channel_idx,
            roi_idx["s_z"].loc[f"{roi}"] : roi_idx["e_z"].loc[f"{roi}"],
            roi_idx["s_y"].loc[f"{roi}"] : roi_idx["e_y"].loc[f"{roi}"],
            roi_idx["s_x"].loc[f"{roi}"] : roi_idx["e_x"].loc[f"{roi}"],
        ] = roi_labels
        roi += 1
        curr_roi_max += roi_max

    label_dir = Path(f"{zarr_url}/labels")
    if label_dir.is_dir() & overwrite:
        shutil.rmtree(label_dir)
    try:
        io.labels_to_ome_zarr(labels, zarr_url, label_name)
    except ContainsArrayError:
        raise FileExistsError(
            f"{zarr_url} already contains labels in the OME-ZARR fileset. To ignore the existing dataset set `overwrite = True`."
        )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=stardist_segmentation,
        logger_name=logger.name,
    )

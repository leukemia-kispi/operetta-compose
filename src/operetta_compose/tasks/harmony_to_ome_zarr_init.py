"""Harmony to OME-Zarr initialization task."""

import logging
from pathlib import Path

from fractal_converters_tools import initiate_ome_zarr_plates
from fractal_converters_tools import AdvancedComputeOptions
from fractal_converters_tools import build_parallelization_list
from pydantic import BaseModel, Field, validate_call
from typing import Any
from operetta_compose.tasks.harmony_to_ome_zarr import _parse_harmony_index
logger = logging.getLogger(__name__)


class AcquisitionInputModel(BaseModel):
    """Acquisition metadata.

    Attributes:
        path: Path to the acquisition directory.
            For scanr, this should include a 'data/' directory with the tiff files
            and a metadata.ome.xml file.
        plate_name: Optional custom name for the plate. If not provided, the name will
            be the acquisition directory name.
        acquisition_id: Acquisition ID,
            used to identify the acquisition in case of multiple acquisitions.
    """

    path: str
    plate_name: str | None = None
    acquisition_id: int = Field(default=0, ge=0)


class ConvertScanrInitArgs(BaseModel):
    """Arguments for the compute task."""

    tiled_image_pickled_path: str
    advanced_options: AdvancedComputeOptions = Field(
        default_factory=AdvancedComputeOptions
    )


@validate_call
def harmony_to_ome_zarr_init(
    *,
    zarr_dir: str,
    img_paths: list[str],
    omero_channels: list[str],
    wavelengths_id: list[str],
    overwrite: bool = False,
    coarsening_xy: int = 2,
    compute: bool = True,
    advanced_options: AdvancedComputeOptions = AdvancedComputeOptions()
) -> dict[str, Any]:
    """
    Convert TIFFs which were exported from Harmony (Operetta/Opera, Perkin-Elmer) to OME-ZARR


    Args:
        zarr_urls: List of zarr urls to be processed (not used by converter task)
        zarr_dir: Path to the new OME-ZARR output directory where the zarr plates should be saved.
            The zarr plates are extracted from the image paths
        img_paths: Paths to the input directories with the image files
        omero_channels: List of Omero channels
        overwrite: Whether to overwrite any existing OME-ZARR directory
        coarsening_xy: Coarsening factor in XY to use for downsampling when building the pyramids
        compute: Wether to compute a numpy array from the dask array while saving the image to the zarr fileset
                 (compute = TRUE is faster given that images fit into memory)
    """
    logging.info(f"{zarr_dir=}")

    image_list_updates = []
    for img_path in img_paths:
        img_path = Path(img_path)
        if img_path.parent.name:
            print(img_path.parent.name)
            plate = img_path.parent.name + ".zarr"
        else:
            logging.info(f"No plate name can be extracted, default to plate.zarr")
            plate = "plate.zarr"
        zarr_path = Path(zarr_dir).joinpath(plate)
        df_wells, df_imgs = _parse_harmony_index(img_path)
        
    """
    create the tiling imagee
    
    """
    tiled_images = []

    parallelization_list = build_parallelization_list(
        zarr_dir=zarr_dir,
        tiled_images=tiled_images,
        overwrite=overwrite,
        advanced_compute_options=advanced_options,
    )
    logger.info(f"Total {len(parallelization_list)} images to convert.")

    initiate_ome_zarr_plates(
        zarr_dir=zarr_dir,
        tiled_images=tiled_images,
        overwrite=overwrite,
    )
    logger.info(f"Initialized OME-Zarr Plate at: {zarr_dir}")
    return {"parallelization_list": parallelization_list}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=harmony_to_ome_zarr_init, logger_name=logger.name)
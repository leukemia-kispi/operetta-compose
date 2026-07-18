"""Detect spots with Spotiflow and count them per segmented object."""

import fcntl
import logging
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from ngio import open_ome_zarr_container
from ngio.experimental.iterators import FeatureExtractorIterator
from ngio.tables import FeatureTable
from ngio.transforms import ZoomTransform
from pydantic import validate_call

logger = logging.getLogger(__name__)


@contextmanager
def _model_download_lock(lock_file: Path):
    """Serialize the Spotiflow model download across parallel tasks.

    Fractal runs one task instance per image, so many processes may try to
    download the same pretrained model at once. An exclusive file lock ensures
    only one of them writes the shared cache at a time.
    """
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _load_spotiflow_model(spotiflow_cls, model_name: str):
    """Load a pretrained Spotiflow model, tolerating a corrupt shared cache.

    Args:
        spotiflow_cls: The ``spotiflow.model.Spotiflow`` class (injected so the
            heavy dependency can stay a lazy import).
        model_name: Name of the pretrained model to load.

    Returns:
        The loaded Spotiflow model.
    """
    cache_dir = Path.home() / ".spotiflow"
    lock_path = Path.home() / ".spotiflow_download.lock"
    with _model_download_lock(lock_path):
        try:
            return spotiflow_cls.from_pretrained(model_name)
        except Exception as err:
            logger.warning(
                f"Loading Spotiflow model '{model_name}' failed ({err}); "
                "clearing the cache and retrying the download."
            )
            shutil.rmtree(cache_dir, ignore_errors=True)
            return spotiflow_cls.from_pretrained(model_name)


def count_spots_per_label(
    points: np.ndarray,
    label_array: np.ndarray,
) -> pd.DataFrame:
    """Assign detected spots to labels and count the spots per label.

    Args:
        points: Spot coordinates with shape ``(n_spots, ndim)`` in the same axis
            order as ``label_array`` (e.g. ``(y, x)`` for a 2D label image).
        label_array: Integer label image where ``0`` marks the background.

    Returns:
        A dataframe indexed by ``label`` with a single ``spot_count`` column.
        Every non-background label is represented, with a count of ``0`` when no
        spot was assigned to it. Spots that fall on the background are ignored.
    """
    labels = np.unique(label_array)
    labels = labels[labels != 0]
    counts = pd.Series(0, index=labels.astype(int), dtype=int)

    points = np.asarray(points)
    if points.size:
        idx = np.floor(points).astype(int)
        # Clip so subpixel detections just outside the image still map inside.
        coords = tuple(
            np.clip(idx[:, dim], 0, label_array.shape[dim] - 1)
            for dim in range(label_array.ndim)
        )
        spot_labels = label_array[coords]
        spot_labels = spot_labels[spot_labels != 0]
        if spot_labels.size:
            counts.update(pd.Series(spot_labels).value_counts())

    counts.index.name = "label"
    return counts.rename("spot_count").to_frame()


@validate_call
def spot_detection(
    *,
    zarr_url: str,
    channel_name: str,
    label_image_name: str,
    roi_table_name: str = "FOV_ROI_table",
    model_name: str = "general",
    output_table_name: str = "spots",
    overwrite: bool = True,
) -> None:
    """Detect spots with Spotiflow and count them per segmented object.

    For every ROI in ``roi_table_name`` the selected channel is run through a
    pretrained Spotiflow model. Each detected spot is assigned to the label it
    falls on and the per-label spot counts are written back as a feature table.

    Requires the optional ``spotiflow`` dependency (``pip install
    "operetta-compose[spotiflow]"``).

    Args:
        zarr_url: Path to an OME-Zarr image (provided by Fractal).
        channel_name: Name of the channel to detect spots in.
        label_image_name: Name of the label image the spots are counted over.
        roi_table_name: Name of the ROI table to iterate over.
        model_name: Name of the pretrained Spotiflow model to use.
        output_table_name: Name of the feature table to write.
        overwrite: Whether to overwrite an existing output table.
    """
    from spotiflow.model import Spotiflow  # lazy: heavy optional dependency

    logger.info(f"Opening {zarr_url}")
    ome_zarr = open_ome_zarr_container(zarr_url)

    if not overwrite and output_table_name in ome_zarr.list_tables():
        raise FileExistsError(
            f"Table '{output_table_name}' already exists. "
            "Set overwrite=True to overwrite it."
        )

    logger.info(f"Loading Spotiflow model '{model_name}'")
    model = _load_spotiflow_model(Spotiflow, model_name)

    image = ome_zarr.get_image()
    label_image = ome_zarr.get_label(
        name=label_image_name, pixel_size=image.pixel_size, strict=False
    )

    # The label image may be stored at a coarser resolution than the intensity
    # image; rescale it (nearest neighbour) so spots and labels share a grid.
    label_zoom_transform = ZoomTransform(
        input_image=label_image,
        target_image=image,
        order="nearest",
    )
    iterator = FeatureExtractorIterator(
        input_image=image,
        input_label=label_image,
        axes_order="yxc",  # force 2D processing (a singleton z is squeezed)
        label_transforms=[label_zoom_transform],
        channel_selection=channel_name,
    )

    try:
        roi_table = ome_zarr.get_table(roi_table_name)
    except KeyError as err:
        raise ValueError(
            f"ROI table '{roi_table_name}' not found in {zarr_url}."
        ) from err
    iterator = iterator.product(roi_table)

    per_roi_counts = []
    for image_slice, label_slice, _roi in iterator.iter_as_numpy():
        points, _details = model.predict(np.squeeze(image_slice))
        per_roi_counts.append(count_spots_per_label(points, np.squeeze(label_slice)))

    if per_roi_counts:
        feature_df = pd.concat(per_roi_counts)
    else:
        feature_df = pd.DataFrame({"spot_count": []}, dtype=int).rename_axis("label")

    feature_table = FeatureTable(
        table_data=feature_df, reference_label=label_image_name
    )
    ome_zarr.add_table(name=output_table_name, table=feature_table, overwrite=overwrite)
    logger.info(
        f"Wrote {len(feature_df)} labels "
        f"({int(feature_df['spot_count'].sum())} spots) to table "
        f"'{output_table_name}'."
    )
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=spot_detection,
        logger_name=logger.name,
    )

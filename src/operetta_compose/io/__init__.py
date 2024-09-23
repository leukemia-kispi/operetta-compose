import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from collections.abc import Iterable
import zarr
from zarr.errors import ArrayNotFoundError
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader, Node
from ome_zarr.writer import (
    write_labels,
)
import dask.array as da
import anndata as ad
from pydantic import BaseModel
from typing import Optional

from fractal_tasks_core.ngff import load_NgffImageMeta


COLORS = ["20adf8", "f8ad20", "942094", "00ffff", "ffff00", "ff00ff", "ffffff"]


class OmeZarrUrl(BaseModel):
    """
    Model for a ZarrUrl
    """

    root: str
    row: Optional[str] = None
    col: Optional[str] = None
    well: Optional[str] = None
    image: Optional[str] = None


def parse_zarr_url(zarr_url: str) -> OmeZarrUrl:
    """Parse the OME-ZARR URL into a dictionary with the root URL, row, column and image

    Args:
        zarr_url: Path to the OME-ZARR

    Returns:
        A `OmeZarrUrl` object
    """
    zarr_dict = {"root": None}
    if zarr_url:
        parts = [p.replace("\\", "") for p in Path(zarr_url).parts]
        for i, p in enumerate(parts):
            if p.endswith(".zarr"):
                zarr_dict["root"] = str(Path(*parts[0 : i + 1]))
                break
        if not zarr_dict["root"]:
            raise ValueError("No .zarr extension detected in URL")
        try:
            zarr_dict["row"] = parts[i + 1]
        except:
            zarr_dict["row"] = None
        try:
            zarr_dict["col"] = parts[i + 2]
            zarr_dict["well"] = zarr_dict["row"] + zarr_dict["col"]
        except:
            zarr_dict["col"] = None
        try:
            zarr_dict["image"] = parts[i + 3]
        except:
            zarr_dict["image"] = None
        return OmeZarrUrl(**zarr_dict)


def read_ome_zarr(zarr_url: Union[str, Path]) -> Node:
    """Read an OME-ZARR fileset

    Args:
        zarr_url: Path to an OME-ZARR

    Returns:
        An ome_zarr image node
    """
    reader = Reader(parse_url(zarr_url))
    zarr_group = list(reader())[0]
    return zarr_group


def convert_ROI_table_to_indices(
    ROI: ad.AnnData,
    pxl_sizes_zyx: Iterable[float],
    cols_xyz_pos: Iterable[str] = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
    ],
    cols_xyz_len: Iterable[str] = [
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ],
) -> dict[str, int]:
    """Convert physical units in region-of-interest tables (ROI) to indices based on the given scale level.

    Args:
        ROI : Name of the region of interest
        pxl_sizes_zyx : Physical size of the zyx pixels in units given defined in multiscales (usually um)
        cols_xyz_pos : Name of columns identifying the xyz positions
        cols_xyz_len : Name of columns identifying the physical dimensions in xyz

    Returns:
        A dictionary with FOV names as keys and a list of starting and end pixel indices as [s_z, e_z, s_y, e_y, s_x, e_x].

    Examples:
        >>> ROI_table = ad.read_zarr("plate.zarr/C/3/0/tables/FOV_ROI_table/")
        >>> operetta_compose.io.convert_ROI_table_to_indices(ROI_table, [1.0, 1.195, 1.195])

    # Note:
    # Modified from https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/main/fractal_tasks_core/roi/v1.py
    """
    pxl_size_z, pxl_size_y, pxl_size_x = pxl_sizes_zyx

    x_pos, y_pos, z_pos = cols_xyz_pos[:]
    x_len, y_len, z_len = cols_xyz_len[:]

    origin_x = min(ROI[:, x_pos].X[:, 0])
    origin_y = min(ROI[:, y_pos].X[:, 0])
    origin_z = min(ROI[:, z_pos].X[:, 0])

    indices_dict = {}
    for FOV in ROI.obs_names:
        x_micrometer = ROI[FOV, x_pos].X[0, 0] - origin_x
        y_micrometer = ROI[FOV, y_pos].X[0, 0] - origin_y
        z_micrometer = ROI[FOV, z_pos].X[0, 0] - origin_z
        len_x_micrometer = ROI[FOV, x_len].X[0, 0]
        len_y_micrometer = ROI[FOV, y_len].X[0, 0]
        len_z_micrometer = ROI[FOV, z_len].X[0, 0]

        start_x = x_micrometer / pxl_size_x
        end_x = (x_micrometer + len_x_micrometer) / pxl_size_x
        start_y = y_micrometer / pxl_size_y
        end_y = (y_micrometer + len_y_micrometer) / pxl_size_y
        start_z = z_micrometer / pxl_size_z
        end_z = (z_micrometer + len_z_micrometer) / pxl_size_z
        indices = list(map(round, [start_z, end_z, start_y, end_y, start_x, end_x]))
        indices_dict[FOV] = indices[:]

    return pd.DataFrame.from_dict(
        indices_dict,
        orient="index",
        columns=["s_z", "e_z", "s_y", "e_y", "s_x", "e_x"],
    )


def get_roi(
    zarr_url: str,
    roi_table: str,
    level: int = 0,
) -> tuple[Path, pd.DataFrame]:
    """Get the zarr path and pixel indices for the selected well at a given resolution level

    Args:
        zarr_url: Path to the OME-ZARR
        roi_table: Name of the ROI table
        level: Resolution level (0 = original, not downsampled resolution level)

    Returns:
        Tuple of zarr url and dataframe with start and end pixel indices (s_z, e_z, s_y, e_y, s_x, e_x)
    """
    roi_tbl = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")

    img_scale = (
        load_NgffImageMeta(f"{zarr_url}")
        .datasets[level]
        .coordinateTransformations[0]
        .scale
    )
    roi_idx = convert_ROI_table_to_indices(roi_tbl, img_scale[-3:])
    roi_url = Path(f"{zarr_url}/{level}")
    return roi_url, roi_idx


def load_intensity_roi(
    roi_url: Path,
    roi_idx: pd.DataFrame,
    roi: int = 0,
    channel: int = 0,
    timepoint: int = 0,
) -> np.ndarray:
    """Load the intensity array of the selected ROI

    Args:
        roi_url: zarr url to the selected ROI
        roi_idx: Dataframe with the x/yZ start and end indices
        roi: Index of the ROI in the selected well
        channel: Channel index
        timepoint: Timepoint index

    Returns:
        Numpy array with the ROI intensities
    """
    try:
        s_z, e_z, s_y, e_y, s_x, e_x = roi_idx.loc[f"{roi}"]
    except KeyError as e:
        raise KeyError(f'ROI named "{roi}" not found in the anndata table.')
    img_data_zyx = da.from_zarr(roi_url)
    if img_data_zyx.ndim == 5:
        img_roi = np.array(img_data_zyx[timepoint, channel, s_z:e_z, s_y:e_y, s_x:e_x])
    elif img_data_zyx.ndim == 4:
        img_roi = np.array(img_data_zyx[channel, s_z:e_z, s_y:e_y, s_x:e_x])
    elif img_data_zyx.ndim == 3:
        img_roi = np.array(img_data_zyx[s_z:e_z, s_y:e_y, s_x:e_x])
    else:
        img_roi = np.array(img_data_zyx[s_y:e_y, s_x:e_x])
    return img_roi


def load_label_roi(
    roi_url: Path,
    roi_idx: pd.DataFrame,
    roi: int = 0,
    name: str = "nuclei",
    channel: int = 0,
    timepoint: int = 0,
) -> np.ndarray:
    """Load the label array of the selected ROI

    Args:
        roi_url: zarr url to the selected ROI
        roi_idx: Dataframe with the x/y/z start and end indices
        roi: ROI index in the selected well
        name: Name of the labels folder
        channel: Channel index
        timepoint: Timepoint index

    Returns:
        Numpy array with the ROI labels
    """
    s_z, e_z, s_y, e_y, s_x, e_x = roi_idx.loc[f"{roi}"]
    try:
        label_data_zyx = da.from_zarr(f"{roi_url.parent}/labels/{name}/{roi_url.name}")
    except ArrayNotFoundError:
        raise FileNotFoundError(
            "No labels exist at the specified zarr URL. Did you run a segmentation?"
        )
    if label_data_zyx.ndim == 5:
        label_roi = np.array(
            label_data_zyx[timepoint, channel, s_z:e_z, s_y:e_y, s_x:e_x]
        )
    elif label_data_zyx.ndim == 4:
        label_roi = np.array(label_data_zyx[channel, s_z:e_z, s_y:e_y, s_x:e_x])
    elif label_data_zyx.ndim == 3:
        label_roi = np.array(label_data_zyx[s_z:e_z, s_y:e_y, s_x:e_x])
    else:
        label_roi = np.array(label_data_zyx[s_y:e_y, s_x:e_x])
    return label_roi


def labels_to_ome_zarr(
    labels: Union[np.ndarray, da.Array],
    zarr_url: str,
    name: str = "nuclei",
):
    """Save labels to the OME-ZARR fileset

    Args:
        labels: Labels array
        zarr_url: Path to the OME-ZARR
        name: Name of the labels folder
    """
    field_group = zarr.group(parse_url(f"{zarr_url}", mode="w").store)
    ds = load_NgffImageMeta(f"{zarr_url}").datasets
    scl_c, scl_z, scl_y, scl_x = ds[0].coordinateTransformations[0].scale
    coarsening_xy = (
        ds[1].coordinateTransformations[0].scale[-1]
        / ds[0].coordinateTransformations[0].scale[-1]
    )
    write_labels(
        labels=np.array(labels),
        group=field_group,
        name=name,
        axes=[
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        coordinate_transformations=[
            [
                {
                    "scale": [
                        scl_c,
                        scl_z,
                        scl_y * coarsening_xy**level,
                        scl_x * coarsening_xy**level,
                    ],
                    "type": "scale",
                }
            ]
            for level in range(5)
        ],
        label_metadata={"source": {"image": "../../"}},
    )


def features_to_ome_zarr(
    zarr_url: str,
    feature_table: pd.DataFrame,
    table_name: str = "regionprops",
    label_name: str = "nuclei",
):
    """Save features to the OME-ZARR fileset

    Args:
        feature_table: Dataframe with feature measurements generated by `feature_table`
        zarr_url: Path to the OME-ZARR
        table_name: Folder name of the measured regionprobs features
        label_name: Name of the labels to use for feature measurements
    """
    ome_zarr_url = parse_zarr_url(zarr_url)
    label = feature_table.pop("label")
    tbl = ad.AnnData(X=feature_table.values)
    tbl.obs = pd.DataFrame({"roi_id": ome_zarr_url.well, "label": label})
    tbl.obs_names = feature_table.index.map(str)
    tbl.var_names = feature_table.columns
    tbl.write_zarr(f"{zarr_url}/tables/{table_name}")
    tables_group = zarr.group(parse_url(f"{zarr_url}/tables", mode="w").store)
    if table_name not in tables_group.attrs["tables"]:
        tables_group.attrs["tables"] = tables_group.attrs["tables"] + [table_name]
    write_table_metadata(zarr_url, "feature_table", table_name, label_name)


def write_table_metadata(
    zarr_url: str, table_type: str, table_name: str, label_name: str = None
):
    """Write table metadata

    Args:
        zarr_url: Path to the OME-ZARR
        table_type: Table type according to the Fractal table spec https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/
        table_name: Folder name of the measured regionprobs features
        label_name: Name of the labels to use for feature measurements
    """
    feature_table_group = zarr.group(
        parse_url(f"{zarr_url}/tables/{table_name}", mode="r+").store
    )
    if table_type == "feature_table":
        attrs = {
            "fractal_table_version": "1",
            "type": "feature_table",
            "region": {"path": f"../labels/{label_name}"},
            "instance_key": "label",
        }
    elif table_type == "roi_table":
        attrs = {"fractal_table_version": "1", "type": "roi_table"}
    elif table_type == "condition_table":
        attrs = {"fractal_table_version": "1", "type": "condition_table"}
    feature_table_group.attrs.update(**attrs)


def condition_to_ome_zarr(
    zarr_url: str,
    condition_table: pd.DataFrame,
    condition_name: str = "condition",
):
    """Save experimental conditions (drug, concentration, etc.) to the OME-ZARR fileset

    Args:
        zarr_url: Path to an OME-ZARR
        condition_table: Dataframe with experimental conditions
        condition_name: Folder name of the experimental condition table
    """
    condition_table.index = condition_table.index.map(str)
    ad_condition = ad.AnnData(X=np.empty((condition_table.shape[0], 0)))
    ad_condition.obs = condition_table
    ad_condition.write_zarr(f"{zarr_url}/tables/{condition_name}")
    table_group = zarr.group(parse_url(f"{zarr_url}/tables", mode="w").store)
    if condition_name not in table_group.attrs["tables"]:
        table_group.attrs["tables"] = table_group.attrs["tables"] + [condition_name]

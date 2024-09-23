import logging
from pathlib import Path
import string
import xmltodict
import pandas as pd
import numpy as np
import zarr
from zarr.errors import (
    ContainsArrayError,
)
from ome_zarr.writer import (
    write_image,
    write_plate_metadata,
    write_well_metadata,
)
from ome_zarr.io import parse_url
import dask.array as da
from dask.array.image import imread
import anndata as ad

from pydantic import validate_call
from typing import Any

import fractal_tasks_core


from operetta_compose import io
from operetta_compose import utils

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

logger = logging.getLogger(__name__)

COLORS = ["20adf8", "f8ad20", "942094", "00ffff", "ffff00", "ff00ff", "ffffff"]


@validate_call
def harmony_to_ome_zarr(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    img_paths: list[str],
    overwrite: bool = False,
    coarsening_xy: int = 2,
    compute: bool = True,
) -> dict[str, Any]:
    """
    Convert TIFFs which were exported from Harmony (Operetta/Opera, Perkin-Elmer) to OME-ZARR


    Args:
        zarr_urls: List of zarr urls to be processed (not used by converter task)
        zarr_dir: Path to the new OME-ZARR output directory where the zarr plates should be saved.
            The zarr plates are extracted from the image paths
        img_paths: Paths to the input directories with the image files
        overwrite: Whether to overwrite any existing OME-ZARR directory
        coarsening_xy: Coarsening factor in XY to use for downsampling when building the pyramids
        compute: Wether to compute a numpy array from the dask array while saving the image to the zarr fileset
                 (compute = TRUE is faster given that images fit into memory)
    """
    logging.info(f"{zarr_dir=}")

    image_list_updates = []
    for img_path in img_paths:
        img_path = Path(img_path)
        plate = img_path.parent.name + ".zarr"
        zarr_path = Path(zarr_dir).joinpath(plate)
        df_wells, df_imgs = _parse_harmony_index(img_path)
        msg = f"Converting Harmony TIFFs from {img_path.parent.name} to OME-ZARR"
        img_list = _create_ome_zarr(
            img_path,
            zarr_path,
            df_wells,
            df_imgs,
            msg,
            overwrite,
            coarsening_xy,
            compute,
        )
        image_list_updates.extend(img_list)

    return {"image_list_updates": image_list_updates}


def _parse_harmony_index(harmony_img_path: Path) -> tuple[pd.DataFrame]:
    """Parses the index files of Harmony (Operetta/Opera, PerkinElmer) and returns two DataFrames wells and images

    Args:
        harmony_img_path: Path to Harmony image folder with index xml and TIFF files

    Returns:
        Tuple of DataFrames df_wells ("row", "col")
        and df_imgs ("row", "col", "pos_x/y/z", "len_x/y/z", "res_x/y/z", "field", "channel", "timepoint", "img_name")
    """
    xml_file = sorted(harmony_img_path.glob("*.xml"))
    if not xml_file:
        raise ValueError(f"Cannot file .xml file in path {harmony_img_path}")
    else:
        xml_file = xml_file[0]
    with open(xml_file, "r") as f:
        idx = xmltodict.parse(f.read(), process_namespaces=False)

    df_wells = (
        pd.DataFrame(idx["EvaluationInputData"]["Wells"]["Well"])
        .astype({"Row": int, "Col": int})
        .rename(columns={"Row": "row", "Col": "col"})[["row", "col"]]
    )
    df_imgs = pd.DataFrame(idx["EvaluationInputData"]["Images"]["Image"]).astype(
        {
            "Row": int,
            "Col": int,
            "TimepointID": int,
            "ChannelID": int,
            "PlaneID": int,
            "FieldID": int,
            "ImageSizeX": int,
            "ImageSizeY": int,
        }
    )

    df_imgs["pos_x"] = (
        pd.DataFrame(
            df_imgs["PositionX"].values.tolist(), index=df_imgs["PositionX"].index
        )["#text"].astype(float)
        * 10**6
    )  # in uM
    df_imgs["pos_y"] = -(
        pd.DataFrame(
            df_imgs["PositionY"].values.tolist(), index=df_imgs["PositionY"].index
        )["#text"].astype(float)
        * 10**6
    )  # in uM => flip axis since Operetta sets the origin at the bottom left wheres napari sets it at the top left
    df_imgs["pos_z"] = (
        pd.DataFrame(
            df_imgs["PositionZ"].values.tolist(), index=df_imgs["PositionZ"].index
        )["#text"].astype(float)
        * 10**6
    )  # in uM
    df_imgs["res_x"] = (
        pd.DataFrame(
            df_imgs["ImageResolutionX"].values.tolist(),
            index=df_imgs["ImageResolutionX"].index,
        )["#text"].astype(float)
        * 10**6
    )  # in uM
    df_imgs["res_y"] = (
        pd.DataFrame(
            df_imgs["ImageResolutionY"].values.tolist(),
            index=df_imgs["ImageResolutionY"].index,
        )["#text"].astype(float)
        * 10**6
    )  # in uM
    try:
        df_imgs["res_z"] = np.diff(df_imgs["pos_z"].unique())[0]  # in uM
    except IndexError:
        df_imgs["res_z"] = 1.0
    df_imgs["len_x"] = df_imgs["ImageSizeX"] * df_imgs["res_x"]
    df_imgs["len_y"] = df_imgs["ImageSizeY"] * df_imgs["res_y"]
    df_imgs["len_z"] = len(df_imgs["PlaneID"].unique()) * df_imgs["res_z"]
    df_imgs = df_imgs.rename(
        columns={
            "Row": "row",
            "Col": "col",
            "FieldID": "field",
            "ChannelID": "channel",
            "TimepointID": "timepoint",
            "URL": "img_name",
        }
    )[
        [
            "row",
            "col",
            "pos_x",
            "pos_y",
            "pos_z",
            "len_x",
            "len_y",
            "len_z",
            "res_x",
            "res_y",
            "res_z",
            "field",
            "channel",
            "timepoint",
            "img_name",
            "ChannelName",
            "MainEmissionWavelength",
        ]
    ]
    return (df_wells, df_imgs)


def _create_ome_zarr(
    img_path: str,
    zarr_url: str,
    df_wells: pd.DataFrame,
    df_imgs: pd.DataFrame,
    msg: str,
    overwrite: bool = False,
    coarsening_xy: int = 2,
    compute: bool = True,
):
    """Initialize an OME ZARR

    Args:
        img_path: Path to the input TIFF image folder
        zarr_url : Path to the output OME-ZARR
        df_wells : DataFrame with integer columns "row" and "col"
        df_imgs : DataFrame with float columns: "pos_x", "pos_y", "pos_z", "len_x", "len_y", "len_z", "res_x", "res_y", "res_z"
            and integer columns "field", "channel", "timepoint"
        msg: Message to display in the progress bar
        overwrite: Whether to overwrite any existing OME-ZARR directory
        coarsening_xy: Coarsening factor in XY to use for downsampling when building the pyramids
        compute: Whether to compute a numpy array from the dask array while saving the image to the zarr fileset
                 (compute = TRUE is faster given that images fit into memory)
    """
    store = parse_url(zarr_url, mode="w").store
    root = zarr.group(store=store, overwrite=overwrite)

    rows = df_wells["row"].unique()
    cols = df_wells["col"].unique()
    dataset = 0

    write_plate_metadata(
        root,
        list(map(lambda row: string.ascii_uppercase[row - 1], rows)),
        list(map(str, cols)),
        [
            f"{string.ascii_uppercase[row-1]}/{col}"
            for row, col in zip(df_wells["row"], df_wells["col"])
        ],
        name=zarr_url.name,
        field_count=df_imgs["field"].max(),
        acquisitions=[{"id": 1, "name": "aq1"}],
    )

    unique_wells = df_wells.drop_duplicates()
    pbar = utils._initialize_pbar(len(unique_wells), msg)
    image_list_updates = list()
    for _, i in unique_wells.iterrows():
        row = i.row
        row_alpha = string.ascii_uppercase[row - 1]
        col = i.col
        well = f"{row_alpha}/{col}"
        well_group = root.require_group(well)
        field_group = well_group.require_group(f"{dataset}")
        table_group = field_group.require_group("tables")

        write_well_metadata(well_group, images=[{"path": str(dataset)}])

        min_pos_z = df_imgs["pos_z"].min()
        min_pos_y = df_imgs["pos_y"].min()
        min_pos_x = df_imgs["pos_x"].min()
        min_channel = df_imgs["channel"].min()
        min_timepoint = df_imgs["timepoint"].min()
        qry = df_imgs.query(
            f"row == {row} & col == {col} & timepoint == {min_timepoint} & channel == {min_channel} & pos_z == {min_pos_z}"
        )
        fov_phys = (
            pd.DataFrame(
                {
                    "x_micrometer": qry["pos_x"] - min_pos_x,
                    "y_micrometer": qry["pos_y"] - min_pos_y,
                    "z_micrometer": qry["pos_z"] - min_pos_z,
                    "len_x_micrometer": qry["len_x"],
                    "len_y_micrometer": qry["len_y"],
                    "len_z_micrometer": qry["len_z"],
                    "x_micrometer_original": qry["pos_x"],
                    "y_micrometer_original": qry["pos_y"],
                    "fov": (qry["field"] - 1).astype(str),
                },
            )
            .set_index(
                "fov",
            )
            .rename_axis(None, axis=0)
        )

        well_phys = {}
        for d in ["x", "y", "z"]:
            min_min_micrometer = fov_phys[f"{d}_micrometer"].min()
            max_max_micrometer = (
                fov_phys[f"{d}_micrometer"].max() + fov_phys[f"len_{d}_micrometer"]
            )
            well_phys[f"{d}_micrometer"] = min_min_micrometer
            well_phys[f"len_{d}_micrometer"] = max_max_micrometer - min_min_micrometer
        well_phys = pd.DataFrame(well_phys, index=["0"])

        ad_fov = ad.AnnData(
            X=fov_phys.values,
            dtype=float,
        )
        ad_fov.obs_names = fov_phys.index
        ad_fov.var_names = fov_phys.columns

        ad_well = ad.AnnData(
            X=well_phys.values,
            dtype=float,
        )
        ad_well.obs_names = well_phys.index
        ad_well.var_names = well_phys.columns

        ad_fov.write_zarr(f"{zarr_url}/{well}/{dataset}/tables/FOV_ROI_table")
        ad_well.write_zarr(f"{zarr_url}/{well}/{dataset}/tables/well_ROI_table")
        table_group.attrs["tables"] = ["FOV_ROI_table", "well_ROI_table"]
        io.write_table_metadata(
            f"{zarr_url}/{well}/{dataset}", "roi_table", "FOV_ROI_table"
        )
        io.write_table_metadata(
            f"{zarr_url}/{well}/{dataset}", "roi_table", "well_ROI_table"
        )

        fovs_pxls = io.convert_ROI_table_to_indices(
            ad_fov,
            df_imgs[["res_z", "res_y", "res_x"]].iloc[0].values,
        )

        qry = df_imgs.query(f"row == {row} & col == {col}")
        timepoints = qry["timepoint"].unique()
        channels = qry["channel"].unique()
        planes = qry["pos_z"].unique()
        img = da.empty(
            (
                len(timepoints),
                len(channels),
                len(planes),
                fovs_pxls["e_y"].max(),
                fovs_pxls["e_x"].max(),
            ),
            dtype=np.uint16,
        )
        for idx_t, timepoint in enumerate(timepoints):
            for idx_c, channel in enumerate(channels):
                for idx_z, pos_z in enumerate(planes):
                    df = df_imgs.query(
                        f"row == {row} & col == {col} & timepoint == {timepoint} & channel == {channel} & pos_z == {pos_z}"
                    )
                    for field in df["field"].values:
                        img_name = df.query(f"field == {field}")["img_name"].iloc[0]
                        fov = fovs_pxls.loc[f"{field - 1}"]
                        img[
                            idx_t,
                            idx_c,
                            idx_z,
                            fov["s_y"] : fov["e_y"],
                            fov["s_x"] : fov["e_x"],
                        ] = imread(img_path.joinpath(img_name).as_posix())[0]

        img = img[0, :, :, :, :]
        try:
            if compute:
                img = np.array(img)
            write_image(
                image=img,
                group=field_group,
                axes=[
                    # {"name": "t", "type": "time", "unit": "second"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                coordinate_transformations=[
                    [
                        {
                            "scale": [
                                # 1.0,
                                1.0,
                                df_imgs.iloc[0]["res_z"],
                                df_imgs.iloc[0]["res_y"] * coarsening_xy**level,
                                df_imgs.iloc[0]["res_x"] * coarsening_xy**level,
                            ],
                            "type": "scale",
                        }
                    ]
                    for level in range(5)
                ],
            )
        except ContainsArrayError:
            raise FileExistsError(
                f"{zarr_url} already contains an OME-ZARR fileset. To ignore the existing dataset set overwrite = True."
            )

        field_group.attrs["omero"] = {
            "channels": [
                {
                    "color": COLORS[channel - 1] if channel < 6 else COLORS[-1],
                    "label": df_imgs.query(f"channel == {channel}")["ChannelName"].iloc[
                        0
                    ],
                    "wavelength_id": df_imgs.query(f"channel == {channel}")[
                        "MainEmissionWavelength"
                    ].iloc[0]["#text"],
                    "active": True,
                }
                for channel in df_imgs["channel"].unique()
            ],
            "pixel_size": {
                "x": df_imgs.iloc[0]["res_x"],
                "y": df_imgs.iloc[0]["res_y"],
                "z": df_imgs.iloc[0]["res_z"],
            },
        }

        attributes = {"plate": str(zarr_url.name), "well": f"{row_alpha}{col}"}

        is_3D = True if len(planes) > 1 else False

        image_list_updates.append(
            {
                "zarr_url": f"{str(zarr_url)}/{well}/{dataset}",
                "attributes": attributes,
                "types": {"is_3D": is_3D},
            }
        )
        pbar.update()

    return image_list_updates


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=harmony_to_ome_zarr,
        logger_name=logger.name,
    )

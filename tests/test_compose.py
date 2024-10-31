import pytest
from pathlib import Path
import zarr

from fractal_tasks_core.channels import ChannelInputModel

from operetta_compose.tasks.harmony_to_ome_zarr import harmony_to_ome_zarr
from operetta_compose.tasks.stardist_segmentation import stardist_segmentation
from operetta_compose.tasks.regionprops_measurement import regionprops_measurement
from operetta_compose.tasks.label_prediction import label_prediction
from operetta_compose.tasks.condition_registration import condition_registration

from operetta_compose.io import OmeroNgffChannel, OmeroNgffWindow

TEST_DIR = Path(__file__).resolve().parent
ZARR_DIR = Path(TEST_DIR).joinpath("test_output")
PLATE = "operetta_plate"
PLATE_ZARR = PLATE + ".zarr"


@pytest.fixture
def _make_output_dir():
    zarr_dir = Path(ZARR_DIR)
    zarr_dir.mkdir(parents=True, exist_ok=True)


def _make_test_zarr(zarr_url):
    table_group = zarr.open(zarr_url, mode="w").require_group("tables")
    table_group.attrs["tables"] = []


@pytest.mark.dependency()
def test_converter(_make_output_dir):
    harmony_to_ome_zarr(
        zarr_urls=[],
        zarr_dir=str(ZARR_DIR),
        img_paths=[str(Path(TEST_DIR).joinpath(PLATE, "Images"))],
        omero_channels=[
            OmeroNgffChannel(
                wavelength_id="525",
                label="CyQuant",
                window=OmeroNgffWindow(start=0, end=20000),
                color="20adf8",
            )
        ],
        overwrite=True,
        compute=True,
    )


@pytest.mark.dependency(depends=["test_converter"])
def test_stardist():
    stardist_segmentation(
        zarr_url=str(ZARR_DIR.joinpath(PLATE_ZARR, "C", "3", "0")),
        channel=ChannelInputModel(label="CyQuant"),
        roi_table="FOV_ROI_table",
        stardist_model="2D_versatile_fluo",
        label_name="nuclei",
        prob_thresh=None,
        nms_thresh=None,
        scale=1,
        level=0,
        overwrite=True,
    )


@pytest.mark.dependency(depends=["test_converter", "test_stardist"])
def test_measure():
    regionprops_measurement(
        zarr_url=str(ZARR_DIR.joinpath(PLATE_ZARR, "C", "3", "0")),
        table_name="regionprops",
        label_name="nuclei",
        level=0,
        overwrite=True,
    )


@pytest.mark.dependency(depends=["test_converter", "test_stardist", "test_measure"])
# @pytest.mark.skip
def test_predict():
    label_prediction(
        zarr_url=str(ZARR_DIR.joinpath(PLATE_ZARR, "C", "3", "0")),
        classifier_path=str(Path(TEST_DIR).joinpath("fixtures", "classifier.pkl")),
        table_name="regionprops",
        label_name="nuclei",
    )


@pytest.mark.dependency(depends=["test_converter"])
def test_register_layout():
    condition_registration(
        zarr_url=str(ZARR_DIR.joinpath(PLATE_ZARR, "C", "3", "0")),
        layout_path=str(Path(TEST_DIR).joinpath("fixtures", "drug_layout.csv")),
        condition_name="condition",
        overwrite=True,
    )


def test_register_layout_utf8_sig():
    zarr_url = ZARR_DIR.joinpath("test_plate.zarr", "B", "03", "0")
    _make_test_zarr(zarr_url)
    condition_registration(
        zarr_url=str(zarr_url),
        layout_path=str(
            Path(TEST_DIR).joinpath("fixtures", "drug_layout_utf8-dom.csv")
        ),
        condition_name="condition",
        overwrite=True,
    )


if __name__ == "__main__":
    pytest.main()

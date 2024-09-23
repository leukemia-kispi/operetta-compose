import pytest
from pathlib import Path

from fractal_tasks_core.channels import ChannelInputModel

from operetta_compose.tasks.harmony_to_ome_zarr import harmony_to_ome_zarr
from operetta_compose.tasks.stardist_segmentation import stardist_segmentation
from operetta_compose.tasks.regionprops_measurement import regionprops_measurement
from operetta_compose.tasks.label_prediction import label_prediction
from operetta_compose.tasks.condition_registration import condition_registration

TEST_DIR = Path(__file__).resolve().parent
ZARR_DIR = Path(TEST_DIR).joinpath("test_output")
PLATE = "operetta_plate"
PLATE_ZARR = PLATE + ".zarr"


@pytest.fixture
def _make_output_dir():
    zarr_dir = Path(ZARR_DIR)
    zarr_dir.mkdir(parents=True, exist_ok=True)


@pytest.mark.dependency()
def test_converter(_make_output_dir):
    harmony_to_ome_zarr(
        zarr_urls=[],
        img_paths=[str(Path(TEST_DIR).joinpath(PLATE, "Images"))],
        zarr_dir=str(ZARR_DIR),
        overwrite=True,
        compute=True,
    )


@pytest.mark.dependency(depends=["test_converter"])
def test_stardist():
    stardist_segmentation(
        zarr_url=str(ZARR_DIR.joinpath(PLATE_ZARR, "C", "3", "0")),
        channel=ChannelInputModel(label="Fluorescein (FITC)"),
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
        classifier_path=str(Path(TEST_DIR).joinpath("classifier.pkl")),
        table_name="regionprops",
        label_name="nuclei",
    )


@pytest.mark.dependency(depends=["test_converter"])
def test_register_layout():
    condition_registration(
        zarr_url=str(ZARR_DIR.joinpath(PLATE_ZARR, "C", "3", "0")),
        layout_path=str(Path(TEST_DIR).joinpath("drug_layout.csv")),
        condition_name="condition",
        overwrite=True,
    )


if __name__ == "__main__":
    pytest.main()

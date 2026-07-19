"""Tests for the count_spots_per_label helper of the spot_detection task."""

import numpy as np

from operetta_compose.tasks.spot_detection import count_spots_per_label


def test_count_spots_per_label():
    # Two labelled regions in a 10x10 image.
    label = np.zeros((10, 10), dtype=int)
    label[0:5, 0:5] = 1
    label[5:10, 5:10] = 2
    # 3 spots on label 1, 1 spot on label 2, 1 spot on the background.
    points = np.array([[1, 1], [2, 2], [3, 3], [7, 7], [0, 9]])

    df = count_spots_per_label(points, label)

    assert df.index.name == "label"
    counts = df["spot_count"].to_dict()
    assert counts[1] == 3
    assert counts[2] == 1
    # The background spot (0, 9) is ignored; only the two labels are reported.
    assert set(counts) == {1, 2}


def test_count_spots_per_label_no_spots():
    # A label with no detected spots must still appear with a count of 0.
    label = np.zeros((6, 6), dtype=int)
    label[0:3, 0:3] = 5
    df = count_spots_per_label(np.empty((0, 2)), label)
    assert df.index.tolist() == [5]
    assert df["spot_count"].tolist() == [0]


def test_count_spots_per_label_clips_out_of_bounds():
    # Subpixel detections just outside the image are clipped back inside.
    label = np.zeros((4, 4), dtype=int)
    label[3, 3] = 7
    points = np.array([[3.9, 3.9]])  # floors to (3, 3) after clipping
    df = count_spots_per_label(points, label)
    assert df.loc[7, "spot_count"] == 1

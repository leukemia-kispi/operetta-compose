"""Convert a napari-feature-classifier classifier into a neutral joblib bundle.

The ``feature_classification`` task no longer depends on napari-feature-classifier
(and therefore no longer pulls in napari). Classifiers trained in the napari
plugin must be converted once into a plain bundle that loads with scikit-learn
alone: a dict with the fitted estimator, the feature names and the class names.

Run this in an environment that has napari-feature-classifier installed, e.g.::

    uvx --with napari-feature-classifier --with joblib --with scikit-learn \\
        python scripts/convert_classifier.py old_classifier.clf classifier.joblib
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd


def convert(src: str, dst: str) -> None:
    """Load a napari-feature-classifier pickle and dump a neutral bundle."""
    clf = pd.read_pickle(src)  # napari_feature_classifier.classifier.Classifier
    bundle = {
        "estimator": clf._classifier,  # fitted sklearn RandomForestClassifier
        "feature_names": list(clf.get_feature_names()),
        "class_names": list(clf._class_names),
    }
    joblib.dump(bundle, dst)
    print(
        f"Wrote {dst}: {len(bundle['feature_names'])} features, "
        f"classes {bundle['class_names']}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("src", help="Input napari-feature-classifier pickle")
    parser.add_argument("dst", help="Output neutral joblib bundle (.joblib)")
    args = parser.parse_args()
    convert(args.src, args.dst)

# operetta-compose <img align="right" height="150" src="https://raw.githubusercontent.com/leukemia-kispi/operetta-compose/master/docs/images/operetta-compose_logo.png">

[![Docs Status](https://github.com/leukemia-kispi/operetta-compose/actions/workflows/build.yml/badge.svg)](https://github.com/leukemia-kispi/operetta-compose/actions/workflows/build_docs.yml)
[![PyPI](https://img.shields.io/pypi/v/operetta-compose)](https://pypi.org/project/operetta-compose/)

[Fractal](https://fractal-analytics-platform.github.io/fractal-tasks-core/) tasks to convert and process images from Perkin-Elmer Opera/Operetta high-content microscopes. Workflows for drug response profiling built upon the OME-ZARR file standard.

## Task library

Currently the following tasks are available:

| Task  | Description |
|---|---|
| condition_registration | Register the experimental layout as a plate-level condition table in the OME-Zarr, aligned with the [fractal-uzh-converters](https://github.com/fractal-analytics-platform/fractal-uzh-converters) Operetta converter |
| feature_classification | Classify cells with a trained scikit-learn classifier (e.g. from the [napari-feature-classifier](https://github.com/fractal-napari-plugins-collection/napari-feature-classifier)) and write them to the OME-Zarr |
| cell_count_aggregation | Aggregate cell counts per experimental condition across a plate and write a plate-level table (and CSV) for drug response profiling |
| spot_detection | Detect spots with [Spotiflow](https://github.com/weigertlab/spotiflow) and count them per segmented object, writing a per-label feature table |

### Migrated (legacy) tasks

The following tasks have been removed from this package. They are superseded by
the maintained, up-to-date implementations linked below (built on modern
[ngio](https://github.com/fractal-analytics-platform/ngio)):

| Removed task | Replacement |
|---|---|
| harmony_to_ome_zarr | [fractal-uzh-converters (Operetta converter)](https://github.com/fractal-analytics-platform/fractal-uzh-converters/tree/main/src/fractal_uzh_converters/operetta) |
| condition_registration (old, per-well) | now part of the [fractal-uzh-converters Operetta converter](https://github.com/fractal-analytics-platform/fractal-uzh-converters/tree/main/src/fractal_uzh_converters/operetta); a plate-level version is kept in this package |
| stardist_segmentation | [fractal-stardist-segmentation-task](https://github.com/fractal-analytics-platform/fractal-stardist-segmentation-task) |
| regionprops_measurement | [`measure_features` in fractal-tasks-core](https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/main/fractal_tasks_core/measure_features.py) |

## Development and installation in Fractal

This project uses [pixi](https://pixi.sh) to manage the environment.

1. Install the environment and the package (editable) with `pixi install`
2. Develop the function according to the [Fractal API](https://fractal-analytics-platform.github.io/version_2/)
3. Run the tests with `pixi run test`
4. Update the image list and the Fractal manifest with `pixi run manifest`
5. Build a wheel file in the `dist` folder of the package with `pixi run build`
6. Collect the tasks on a Fractal server

`feature_classification` runs a trained scikit-learn classifier loaded from a neutral joblib
bundle. Classifiers trained in the [napari-feature-classifier](https://github.com/fractal-napari-plugins-collection/napari-feature-classifier)
must be converted once with `python scripts/convert_classifier.py <classifier.clf> classifier.joblib`
(run in an environment that has napari-feature-classifier installed).

Some tasks pull in heavy, task-specific dependencies that are shipped as optional extras:

- `spot_detection` needs Spotiflow (and torch): `pip install -e ".[spotiflow]"`


## Updating docs

1. Update the documentation under `/docs`
2. Update the function API with `quartodoc build`
3. Preview the documentation with `quarto preview`

---

[Fractal](https://fractal-analytics-platform.github.io/fractal-tasks-core/) is developed by the [UZH BioVisionCenter](https://www.biovisioncenter.uzh.ch/de.html) under the lead of [@jluethi](https://github.com/jluethi) and under contract with [eXact lab S.r.l.](https://www.exact-lab.it).

# operetta-compose <img align="right" height="150" src="https://raw.githubusercontent.com/leukemia-kispi/operetta-compose/master/docs/images/operetta-compose_logo.png">

[![Docs Status](https://github.com/leukemia-kispi/operetta-compose/actions/workflows/build.yml/badge.svg)](https://github.com/leukemia-kispi/operetta-compose/actions/workflows/build_docs.yml)
[![PyPI](https://img.shields.io/pypi/v/operetta-compose)](https://pypi.org/project/operetta-compose/)

[Fractal](https://fractal-analytics-platform.github.io/fractal-tasks-core/) tasks to convert and process images from Perkin-Elmer Opera/Operetta high-content microscopes. Workflows for drug response profiling built upon the OME-ZARR file standard.

## Task library

Currently the following tasks are available:

| Task  | Description |
|---|---|
| harmony_to_ome_zarr | Convert TIFFs which were exported from Harmony (Operetta/Opera, Perkin-Elmer) to OME-ZARR |
| stardist_segmentation | Segment cells with Stardist |
| regionprops_measurement | Take measurements using regionprops and write the features to the OME-ZARR |
| label_prediction | Make predictions on the selected wells and write them to the OME-ZARR |
| condition_registration | Register the experimental conditions in the OME-ZARR |

## Development and installation in Fractal

1. Install the package in dev mode with `python -m pip install -e ".[dev]"`
2. Develop the function according to the [Fractal API](https://fractal-analytics-platform.github.io/version_2/)
3. Update the image list and the Fractal manifest with `python src/operio_fractal/dev/create_manifest.py`
4. Build a wheel file in the `dist` folder of the package with `python -m build`
5. Collect the tasks on a Fractal server


## Updating docs

1. Update the documentation under `/docs`
2. Update the function API with `quartodoc build`
3. Preview the documentation with `quarto preview`

---

[Fractal](https://fractal-analytics-platform.github.io/fractal-tasks-core/) is developed by the [UZH BioVisionCenter](https://www.biovisioncenter.uzh.ch/de.html) under the lead of [@jluethi](https://github.com/jluethi) and under contract with [eXact lab S.r.l.](https://www.exact-lab.it).

from fractal_task_tools.task_models import ConverterNonParallelTask, ConverterCompoundTask, ParallelTask

AUTHORS="Fabio Steffen"
DOCS_LINK = "https://leukemia-kispi.github.io/operetta-compose/"
INPUT_MODELS = [
    ("operetta_compose.io", "__init__.py", "OmeroNgffWindow"),
    ("operetta_compose.io", "__init__.py", "OmeroNgffChannel"),
    (
        "operetta_compose",
        "tasks/harmony_to_ome_zarr_init.py",
        "AcquisitionInputModel",
    ),
    #(
    #    "fractal_converters_tools",
    #    "__init__.py",
    #    "AdvancedComputeOptions",
    #),
]

TASK_LIST = [
    ConverterNonParallelTask(
        name="Harmony to OME-Zarr",
        executable="tasks/harmony_to_ome_zarr.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Conversion",
        modality="HCS",
        tags=["Opera", "Operetta", "Perkin Elmer"],
    ),
    ConverterCompoundTask(
        name="Convert Harmony to OME-Zarr V2",
        executable_init="tasks/harmony_to_ome_zarr_init.py",
        executable="tasks/harmony_to_ome_zarr_compute.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 12000},
        category="Conversion",
        modality="HCS",
        tags=["Opera", "Operetta", "Perkin Elmer"],
        # docs_info="file:docs_info/scanr_task.md",
    ),
    ParallelTask(
        name="Stardist segmentation",
        executable="tasks/stardist_segmentation.py",
        meta={"cpus_per_task": 4, "mem": 16000, "needs_gpu": True},
        category="Segmentation",
    ),
    ParallelTask(
        name="Regionprops measurement",
        executable="tasks/regionprops_measurement.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Measurement",
        tags=["regionprops", "intensity", "morphology"],
    ),
    ParallelTask(
        name="Feature classification",
        executable="tasks/feature_classification.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        tags=["napari feature classifier", "object classification"],
    ),
    ParallelTask(
        name="Condition registration",
        executable="tasks/condition_registration.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        modality="HCS",
        tags=["metadata", "well conditions", "perturbation", "treatment"],
    ),
]
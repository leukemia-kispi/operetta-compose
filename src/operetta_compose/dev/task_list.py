from fractal_tasks_core.dev.task_models import NonParallelTask, ParallelTask

TASK_LIST = [
    NonParallelTask(
        name="Harmony to OME-Zarr",
        executable="tasks/harmony_to_ome_zarr.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Conversion",
        modality="HCS",
        tags=["Opera", "Operetta", "Perkin Elmer"],
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

from fractal_tasks_core.dev.task_models import NonParallelTask, ParallelTask

TASK_LIST = [
    NonParallelTask(
        name="Harmony to OME-Zarr",
        executable="tasks/harmony_to_ome_zarr.py",
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
    ParallelTask(
        name="Stardist segmentation",
        executable="tasks/stardist_segmentation.py",
        meta={"cpus_per_task": 4, "mem": 16000},
    ),
    ParallelTask(
        name="Regionprops measurement",
        executable="tasks/regionprops_measurement.py",
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
    ParallelTask(
        name="Label prediction",
        executable="tasks/label_prediction.py",
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
    ParallelTask(
        name="Condition registration",
        executable="tasks/condition_registration.py",
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
]

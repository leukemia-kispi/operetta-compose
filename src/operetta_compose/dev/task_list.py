from fractal_task_tools.task_models import NonParallelTask, ParallelTask

AUTHORS = "Fabio Steffen"
DOCS_LINK = "https://leukemia-kispi.github.io/operetta-compose/"
INPUT_MODELS = []


TASK_LIST = [
    ParallelTask(
        name="Feature classification",
        executable="tasks/feature_classification.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Measurement",
        tags=["napari feature classifier", "object classification"],
    ),
    NonParallelTask(
        name="Condition registration",
        executable="tasks/condition_registration.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        modality="HCS",
        tags=["metadata", "well conditions", "perturbation", "treatment"],
    ),
    NonParallelTask(
        name="Cell count aggregation",
        executable="tasks/cell_count_aggregation.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        modality="HCS",
        category="Measurement",
        tags=["drug response profiling", "cell count", "aggregation"],
    ),
    ParallelTask(
        name="Spot detection",
        executable="tasks/spot_detection.py",
        meta={"cpus_per_task": 1, "mem": 8000},
        category="Measurement",
        tags=["spotiflow", "spot detection", "object features"],
    ),
]

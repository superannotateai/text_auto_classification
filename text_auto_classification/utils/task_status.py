import dataclasses
from enum import Enum


class TaskStatus(str, Enum):
    NOT_STARTED = "Not started"
    DOWNLOADING_DATA = "Downloading data"
    MODEL_TRAINING = "Model training"
    COMPLETED = "Completed"


@dataclasses.dataclass
class TaskInfo():
    status: TaskStatus = TaskStatus.NOT_STARTED
    progress_value: float = 0 # TODO calculate progress value

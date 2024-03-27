from fastapi import APIRouter, BackgroundTasks, Request, status
from starlette.responses import JSONResponse

from text_auto_classification.utils.task_status import TaskStatus

router = APIRouter()


@router.post(
    "/train_predict",
    response_model=None,
    status_code=status.HTTP_200_OK,
    description="Train model on annotated data from SA project and auto annotate other data"
)
def train_predict(
    background_tasks: BackgroundTasks,
    meta: Request
):
    
    task_info = meta.app.state.task
    if task_info.status != TaskStatus.NOT_STARTED and task_info.status != TaskStatus.COMPLETED:
        return JSONResponse("Training is already started", 429)
    
    current_app = meta.app
    classifier = current_app.classifier

    background_tasks.add_task(classifier.train_predict, task_info)

    return JSONResponse("Process started", 200)

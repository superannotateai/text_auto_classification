from fastapi import APIRouter, Request, status
from starlette.responses import JSONResponse

router = APIRouter()

@router.get(
    "/status",
    response_model=None,
    status_code=status.HTTP_200_OK,
    description="Method for status tracking"
)
def status(meta: Request):
    task_info = meta.app.state.task
    return JSONResponse({"status": task_info.status.value}, 200)

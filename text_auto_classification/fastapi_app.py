import argparse
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from text_auto_classification.controllers.ping import router as health_router
from text_auto_classification.controllers.status import router as status_router
from text_auto_classification.controllers.train_predict import \
    router as train_predict_router
from text_auto_classification.utils.auto_classifier import SAAutoClassifier
from text_auto_classification.utils.task_status import TaskInfo


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load indexes
    app.state.task = TaskInfo()
    yield

with open("./version.txt") as f:
    version = f.read()

app = FastAPI(
    title='Text Auto Classification',
    description='Service for train classification model with data labled on SuperAnnotate',
    version=version,
    root_path="/text-auto-classification",
    lifespan=lifespan
)

logging.basicConfig(level=logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(train_predict_router)
app.include_router(status_router)


def parse_args():
    DEFAULT_SERVICE_CONFIG = "etc/configs/service_config.json"
    DEFAULT_TRAINING_CONFIG = "etc/configs/train_config.json"
    DEFAULT_DEVICE = "cuda:0"
    DEFAULT_HOST = "0.0.0.0"
    DEFAULT_PORT = "8080"
    
    parser = argparse.ArgumentParser(
        description="SuperAnnotate service for Text Auto-Classification"
    )
    parser.add_argument(
        "--service-config-path",
        "-sc",
        help=f"Path to the service config (defult: {DEFAULT_SERVICE_CONFIG})",
        default=DEFAULT_SERVICE_CONFIG,
        type=check_path,
    )
    parser.add_argument(
        "--training-config-path",
        "-tc",
        help=f"Path to the training config (defult: {DEFAULT_TRAINING_CONFIG})",
        default=DEFAULT_TRAINING_CONFIG,
        type=check_path,
    )
    parser.add_argument(
        "--device",
        "-d",
        help=f"Device for training and inference model.",
        default=DEFAULT_DEVICE,
        type=str,
    )
    parser.add_argument(
        "--host",
        help=f"Host of service (default: {DEFAULT_HOST})",
        default=DEFAULT_HOST,
        type=str,
    )
    parser.add_argument(
        "--port",
        "-p",
        help=f"Port of service (default: {DEFAULT_PORT})",
        default=DEFAULT_PORT,
        type=int,
    )
    
    return parser.parse_args()


def check_path(path: str) -> str|None:
    
    if not isinstance(path, str):
        raise TypeError("The variable is not a string (path).")

    # Check if the path exists
    if not os.path.exists(path):
        raise TypeError(f"The path '{path}' does not exist.")

    return path


def create_app(path_to_service_conf: os.PathLike, path_to_training_conf: os.PathLike):
    application = app

    with open(path_to_service_conf, 'r') as f:
        service_conf = json.load(f)

    with open(path_to_training_conf, 'r') as f:
        training_conf = json.load(f)

    classifier = SAAutoClassifier(service_conf, training_conf)
    setattr(application, "classifier", classifier)

    class EndpointFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.getMessage().find(f"/segmentation/healthcheck") == -1

    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

    return application


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    # TODO Use ``args.device``
    app = create_app(args.service_config_path, args.training_config_path)
    uvicorn.run(app, host=args.host, port=args.port)
else:
    app = create_app(os.environ.get("DEFAULT_SERVICE_CONFIG"), os.environ.get("DEFAULT_TRAINING_CONFIG"))

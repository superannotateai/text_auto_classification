import json
import os
import urllib.parse
from datetime import datetime
from time import sleep, time

import requests
from superannotate import SAClient

SA_TOKEN = os.environ["SA_TOKEN"]
URL = os.environ["URL"]
# Constant for limiting the amount of data for starting Auto-Classification
# You can change it, but by default it's set up to 100, changing the limit to less may lead to unstable results
COUNT_ITEMS_PER_CLASS = 100

sa = SAClient(token=SA_TOKEN)


def read_status(resp):
    return json.loads(resp.content.decode()).get("status")


def check_enough_data(project_name, threshold):
    project_metadata = sa.get_project_metadata(
        project = project_name,
        include_annotation_classes=True
    )

    classes = [cl["name"] for cl in project_metadata["classes"] if cl["type"] == "tag"]

    enough_data_flag = True
    for cl in classes:
        cl_items = sa.query(
            project = project_name,
            query = f"metadata(status =Completed) AND instance(className = {cl})"
        )

        if len(cl_items) < threshold:
            print(f"Amount of completed items is too small for *{cl}*. {len(cl_items)}/{threshold}")
            enough_data_flag = False

    return enough_data_flag


def handler(event, context):
    # Get project name
    project_name = sa.get_project_by_id(context['after']['project_id'])['name']
    
    # Can't run service if count completed items less than COUNT_ITEMS_PER_CLASS per class
    if not check_enough_data(project_name, COUNT_ITEMS_PER_CLASS):
        return False
    
    # Call serice
    started = start_train_predict()
    if not started:
        return False
    
    # Loop of monitoring the service and waiting for execution
    while True:
        resp = requests.get(urllib.parse.urljoin(URL, "text-auto-classification/status"))

        print(f"Status code: {read_status(resp)}, waiting")
        # Create datetime object from current timestamp
        dt = datetime.fromtimestamp(int(time()))
        # Format datetime as "YYYY-MM-DD hh:mm:ss"
        formatted_datetime = dt.strftime("%Y-%m-%d %H:%M:%S")
        print(formatted_datetime)

        if resp.status_code == 200 and read_status(resp) == "Completed":
            return True
        if (resp.status_code == 200 and read_status(resp) == "Failed") or resp.status_code != 200:
            print(resp.status_code)
            print(read_status(resp))
            return False
        
        sleep(60)


def start_train_predict():
    resp = requests.post(urllib.parse.urljoin(URL, "text-auto-classification/train_predict"))
    
    if resp.status_code == 200:
        return True
    else:
        print(resp.status_code)
        print(read_status(resp))
        return False

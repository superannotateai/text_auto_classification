import json
import logging
import os
import shutil
import time

from superannotate import SAClient

logger = logging.getLogger(__name__)


class SAClientManager():
    """A class used to work with SuperAnnotate platform.
    Download and upload annotations.

    :param config_path: Path to config file for authorizing to SA SDK.
        https://doc.superannotate.com/docs/python-sdk#initialization-and-authorization
    :type config_path: str
    :param project_name: The name of the working project
    :type project_name: str
    :param folders: List of folder to work with within the project. If the value is None, we work with documents only in the project root
    :type folders: list[str] | None
    """
    def __init__(self,
        config_path: str,
        project_name: str,
        folders: list[str] | None   
    ) -> None:
        """Constructor method
        """
        self.sa_client = SAClient(config_path=config_path)
        self.project_name = project_name
        self.folders = folders


    def download_annotations(self) -> list[dict]:
        """Method for downloading annotations locally

        :return: Annotations in the format of a list of dictionaries, where each dictionary represents one record.
            The record is expected to contain following keys: ``folder``, ``item_id``, ``label``
        :rtype: list[dict]
        """
        # Dowload to local storage
        path_to_annotations = "data/annotations"

        if not os.path.exists(path_to_annotations):
            os.makedirs(path_to_annotations)

        annotations = []
        for attempt in range(5):

            self.sa_client.prepare_export(
                project=self.project_name,
                folder_names=self.folders if self.folders else ["root"]
            )
            exports = self.sa_client.get_exports(project=self.project_name)

            self.sa_client.download_export(
                project=self.project_name,
                export=exports[0],
                folder_path=path_to_annotations
            )

            # Read local saved files and transform
            temp_path_not_found = None

            for folder in self.folders if self.folders else [None]:
                
                temp_path = os.path.join(path_to_annotations, folder) \
                    if folder else path_to_annotations
                
                if not os.path.exists(temp_path):
                    temp_path_not_found = temp_path
                    break
                
                for file in os.listdir(temp_path):

                    annotation_path = os.path.join(os.path.join(temp_path, file))
                    with open(annotation_path) as f:
                        annotation = json.load(f)
                    
                    labels = [
                        instance["className"] for instance in annotation["instances"]
                        if instance["type"] == "tag"
                    ]

                    record = {
                        "folder": folder,
                        "item_id": annotation["metadata"]["name"],
                        "label": labels if labels else None,
                        "status": annotation["metadata"]["status"]
                    }

                    annotations.append(record)
                
            if temp_path_not_found is None:
                break
            else:
                annotations = []
                logger.warning(f"Failed to prepare an export. Attempt {attempt} out of 5. Path not found: {temp_path_not_found}")
                time.sleep(60)
            
        # Remove all temporary data
        shutil.rmtree(path_to_annotations)
        
        if not annotations:
            raise Exception("Failed to prepare an export")
        
        return annotations
        

    def upload_annotations(self, dataset: list[dict]) -> None:
        """Method for uploading annotations to platform

        :param dataset: Dataset for uploading in the format of a list of dictionaries, where each dictionary represents one record.
            The record is expected to contain at least the following keys: ``folder``, ``item_id``, ``label``, ``score``
        :type dataset: list[dict]
        """
        for folder in self.folders if self.folders else [None]:

            temp_annotations, items = [], []
            for record in dataset:
                if record["folder"] != folder:
                    continue
                
                items.append(record["item_id"])
                temp_annotations.append({
                    "metadata": {
                        "name": record["item_id"]
                    },
                    "instances": [{
                        "type": "tag",
                        "className": record["label"],
                        "probabylity": record["score"]*100 # Scale scores from 0 to 100
                    }]
                })

            temp_path = os.path.join(self.project_name, folder) \
                if folder else self.project_name

            # Upload new annotation to platform
            self.sa_client.upload_annotations(
                project=temp_path,
                annotations=temp_annotations
            )

            # Change all uploaded itmes status to `QualityCheck`
            self.sa_client.set_annotation_statuses(
                project=temp_path,
                annotation_status="QualityCheck",
                items=items
            )


import logging
import os
import time

from superannotate import SAClient

logger = logging.getLogger("uvicorn")


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
        annotations = []
        for attempt in range(5):
            
            for folder in self.folders if self.folders else [None]:
                
                temp_path = os.path.join(self.project_name, folder) \
                    if folder else self.project_name
                
                temp_items = self.sa_client.search_items(
                    project=temp_path,
                )
                
                temp_annotations = self.sa_client.get_annotations(
                    project=temp_path, 
                    items=[item["name"] for item in temp_items]
                )

                for ann in temp_annotations:
                    
                    labels = [
                        instance["className"] for instance in ann["instances"]
                        if instance["type"] == "tag"
                    ]

                    record = {
                        "folder": folder,
                        "item_id": ann["metadata"]["name"],
                        "label": labels if labels else None,
                        "status": ann["metadata"]["status"]
                    }

                    annotations.append(record)
                
            if annotations:
                break
            else:
                logger.warning(f"Failed to downloading annotations. Attempt {attempt} out of 5.")
                time.sleep(60)
        
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

            temp_annotations = []
            for record in dataset:
                if record["folder"] != folder:
                    continue
                
                temp_annotations.append({
                    "metadata": {
                        "name": record["item_id"]
                    },
                    "instances": [{
                        "type": "tag",
                        "className": record["label"],
                        "probability": record["score"]*100 # Scale scores from 0 to 100
                    }]
                })

            temp_path = os.path.join(self.project_name, folder) \
                if folder else self.project_name

            # Upload new annotation to platform
            res = self.sa_client.upload_annotations(
                project=temp_path,
                annotations=temp_annotations
            )

            logger.info(f"Result of uploading annotations:\n{res}")

            # Change statuses
            if res.get("succeeded"):
                # Find skipped items and assign them accordingly status
                not_started_item_names = [
                    item['name'] for item in self.sa_client.search_items(
                        project=temp_path,
                        annotation_status="NotStarted"
                    )
                ]
                skipped_item_names = [item for item in not_started_item_names if item not in res]
                
                self.sa_client.set_annotation_statuses(
                    project=temp_path,
                    annotation_status="Skipped",
                    items=skipped_item_names
                )

                # Assign all uploaded itmes status to `QualityCheck`
                self.sa_client.set_annotation_statuses(
                    project=temp_path,
                    annotation_status="QualityCheck",
                    items=res.get("succeeded")
                )
            else:
                logger.warning("Predicted annotations weren't uploaded to the platform.")

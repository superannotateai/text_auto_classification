import os
from urllib.parse import urlparse

from botocore.client import BaseClient

from text_auto_classification.utils.data.data_loader import DataLoader


class S3DataLoader(DataLoader):
    """Class for load original text of documents from S3 bucket

    :param s3_client: S3 client from boto lybrary as SDK for S3
    :type s3_client: botocore.client.BaseClient
    :param aws_url_for_data_downloads: The S3 URL to the root of folder with data.
        URL usually contain bucket name and path, looks like `s3://[bucket_name]/[path]` 
    :type aws_url_for_data_downloads: str
    """
    def __init__(self, s3_client: BaseClient, aws_url_for_data_downloads: str) -> None:
        self.s3_client = s3_client

        # Parse aws URL to extract bcuket name and specific path
        parsed_url = urlparse(aws_url_for_data_downloads, allow_fragments=False)
        self.s3_bucket = parsed_url.netloc
        self.s3_path = parsed_url.path.strip("/")


    def load_document(self, item_id: str, folder: str|None = None) -> str:
        """Load original text of document

        :param item_id: Name/Id of item on SA platform. It is expected that on S3 it has the same name.
        :type item_id: str
        :param folder: Name of folder on SA platform. It is expected that on S3 it has the same path,
            defaults to None
        :type folder: str|None
        :return: Text of document
        :rtype: str
        """
        if folder:
            path_to_file = os.path.join(self.s3_path, folder, item_id)
        else:
            path_to_file = os.path.join(self.s3_path, item_id)
        
        response = self.s3_client.get_object(
            Bucket=self.s3_bucket, 
            Key=path_to_file
        )

        text = response['Body'].read().decode('utf-8')

        return text

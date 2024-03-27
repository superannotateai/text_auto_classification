import logging
import os
from urllib.parse import urlparse

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from text_auto_classification.utils.model.model_saver import ModelSaver


class S3ModelSaver(ModelSaver):
    """Class for saving model and tokenizer checkpoints to S3 bucket
    """
    def __init__(self, s3_client, aws_url_to_model_upload) -> None:
        """Constructor method
        """
        self.s3_client = s3_client

        # Parse aws URL to extract bcuket name and specific path
        parsed_url = urlparse(aws_url_to_model_upload, allow_fragments=False)
        self.s3_bucket = parsed_url.netloc
        self.s3_path = parsed_url.path.strip("/")


    def save(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer) -> None:
        """Method for saving checkpoints

        :param model: Model for saving
        :type model: AutoModelForSequenceClassification
        :param tokenizer: Tokenizer for saving
        :type tokenizer: AutoTokenizer
        """
        # Save tokenizer and model locally
        path_to_checkpoints = "data/checkpoints"
        tokenizer.save_pretrained(path_to_checkpoints)
        model.save_pretrained(path_to_checkpoints)

        # Load all checkpoints to S3 bucket
        for file in os.listdir(path_to_checkpoints):
            self.s3_client.upload_file(
                os.path.join(path_to_checkpoints, file),
                self.s3_bucket,
                os.path.join(self.s3_path, file)
            )
        
        logging.info(f"Model and tokenizer were saved successfully in S3 by following path: {os.path.join(self.s3_bucket, self.s3_path)}")

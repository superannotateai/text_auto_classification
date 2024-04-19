import json
import logging
import shutil
from typing import Callable

import boto3
import numpy as np
from datasets import ClassLabel, Dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline
)

from text_auto_classification.utils.data.data_processing import preprocessing
from text_auto_classification.utils.data.s3_data_loader import S3DataLoader
from text_auto_classification.utils.data.sa_client_manager import SAClientManager
from text_auto_classification.utils.model.s3_model_saver import S3ModelSaver
from text_auto_classification.utils.task_status import TaskInfo, TaskStatus

logger = logging.getLogger("uvicorn")


class SAAutoClassifier():
    """Main class of Auto Text Classification
    
    :param service_config: Service configuration containing SA configuration path, project name, and folders.
    :type service_config: dict
    :param training_config: Training configuration containing parameters for training.
    :type training_config: dict
    """
    def __init__(self, service_config: dict, training_config: dict):
        """Constructor method
        """
        self.sa_clinet_manager = SAClientManager(
            service_config["SA_CONFIG_PATH"],
            service_config["SA_PROJECT_NAME"],
            service_config["SA_FOLDERS"]
        )

        s3_client = boto3.client(
            "s3",
            aws_access_key_id=service_config["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=service_config["AWS_SECRET_ACCESS_KEY"]
        )

        self.s3_data_loader = S3DataLoader(
            s3_client,
            service_config["AWS_URL_FOR_DATA_DOWNLOADS"]
        )

        self.s3_model_saver = S3ModelSaver(
            s3_client,
            service_config["AWS_URL_TO_MODEL_UPLOAD"]
        )

        self.training_config = training_config

        self.preprocessing_func = preprocessing


    def _download_and_prepare_data(self) -> list[dict]:
        """Method for downloading and preparation data
        Download annotation from paltform and specific folders by SAClientManager
        Download origin documents from integration by S3DataLoader
        And preprocess data by preprocessing_func
        
        :return: Processed data ready for training.
        :rtype: list[dict]
        """
        annotations = self.sa_clinet_manager.download_annotations()
        
        logger.info("Read documents from S3")
        for record in tqdm(annotations):
            record.update({
                "text": self.s3_data_loader.load_document(
                    record["item_id"],
                    record["folder"]
                )
            })
        data = annotations

        data = self.preprocessing_func(data)

        return data


    def train_predict(self, task_info: TaskInfo):
        """Main method that will be called externally
        Downloading and processing data, trains the model, performs predictions and upload to platform.
        
        :param task_info: Task information object.
        :type task_info: TaskInfo
        """
        try:
            logger.info("Start Auto-Classification Pipeline")
                
            
            task_info.status = TaskStatus.DOWNLOADING_DATA # Update the task status
            logger.info("Downloading and prcessing data")

            data = self._download_and_prepare_data()

            train_data = [rec for rec in data if rec["status"] == "Completed"]
            data_for_predict = [rec for rec in data if rec["status"] != "Completed"]

            logger.info(f"{len(train_data)} completed items were downloading")
            logger.info(f"{len(data_for_predict)} items for prediction were downloading")
            
            # Extract classes
            classes = list({lab for rec in train_data for lab in rec["label"]})
            logger.info(f"The output layer will have {len(classes)} outputs, for each of the annotated classes:\n{classes}")

            # Change labels to ids
            lab2id = {lab:i for i, lab in enumerate(classes)}
            train_data = list(map(lambda item: {**item, "label": lab2id[item["label"][0]]}, train_data))

            # Init dataset
            dataset = Dataset.from_list(train_data)

            # Setup special type for labels
            new_features = dataset.features.copy()
            new_features["label"] = ClassLabel(names=classes)
            dataset = dataset.cast(new_features)

            # Split into train and validation
            dataset = dataset.train_test_split(
                test_size=self.training_config["validation_ratio"],
                stratify_by_column="label",
                seed=19
            )

            # Init tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.training_config["pretrain_model"])
            tokenize = lambda text: tokenizer.batch_encode_plus(
                text,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=self.training_config["max_length"],
                return_tensors="pt"
            )

            tokenized_dataset = dataset.map(lambda x: tokenize(x["text"]), batched=True)
            
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=self.training_config["pretrain_model"],
                problem_type="single_label_classification",
                num_labels=len(classes),
                id2label={v:k for k,v in lab2id.items()},
                label2id=lab2id
            )

            training_args = TrainingArguments(
                output_dir="output_dir",
                evaluation_strategy="epoch",
                per_device_train_batch_size=self.training_config["per_device_train_batch_size"],
                per_device_eval_batch_size=self.training_config["per_device_eval_batch_size"],
                gradient_accumulation_steps=self.training_config["gradient_accumulation_steps"],
                learning_rate=self.training_config["learning_rate"],
                weight_decay=self.training_config["weight_decay"],
                num_train_epochs=self.training_config["num_train_epochs"],
                lr_scheduler_type=self.training_config["lr_scheduler_type"],
                warmup_ratio=self.training_config["warmup_ratio"],
                logging_steps=100,
                save_strategy="epoch",
                save_total_limit=3,
                load_best_model_at_end=True,
                optim=self.training_config["optim"],
                gradient_checkpointing=True
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["test"],
                compute_metrics=self.prepare_compute_metrics_func(len(classes))
            )

            task_info.status = TaskStatus.MODEL_TRAINING # Update the task status
            logger.info("Start Training")

            trainer.train()

            task_info.status = TaskStatus.PREDICTING # Update the task status
            logger.info("Start prediction stage")
            if data_for_predict:
                predict_data = self._predict(
                    tokenizer,
                    model,
                    data_for_predict
                )
                self.sa_clinet_manager.upload_annotations(predict_data)

            self.s3_model_saver.save(model, tokenizer)

            # Remove all temporary data
            shutil.rmtree("output_dir")

            task_info.status = TaskStatus.COMPLETED # Update the task status
            logger.info("Finish full Pipeline")

        except Exception as e:
            task_info.status = TaskStatus.FAILED # Update the task status
            logger.exception("Failed during pipeline")


    def _predict(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForSequenceClassification,
        predict_data: list[dict]
    ) -> list[dict]:
        """Prediction method.
        Take for input data and annotate it by model predictions.
        
        :param tokenizer: Pre-trained Tokenizer.
        :type tokenizer: AutoTokenizer
        :param model: Fine-tuned model
        :type model: AutoModelForSequenceClassification.
        :param predict_data: List of records for prediction.
        :type predict_data: list[dict]
        :return: Predicted data.
        :rtype: list[dict]
        """
        classifier = pipeline(
            "text-classification",
            tokenizer=tokenizer,
            model=model,
            device=0
        )

        predicts = classifier([rec["text"] for rec in predict_data])

        for predict, record in zip(predicts, predict_data):
            record.update(
                {
                    "label": predict["label"],
                    "score": predict["score"]
                }
            )

        return predict_data
    

    @staticmethod
    def prepare_compute_metrics_func(count_classes: int) -> Callable:
        """Static method for creation compute metrics functions
            
        :param count_classes: Count classes.
        :type count_classes: int
        :return: Function for calculating metric.
        :rtype: Callable
        """
        if count_classes == 2:
            average_param_f1 = "binary"
        else:
            average_param_f1 = "micro"
        
        def compute_metrics(eval_pred) -> dict:
            """Function for calculating metric
            
            :param eval_pred: Evaluation prediction.
            :return: Dictionary containing calculated metrics.
            :rtype: dict
            """
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average=average_param_f1)
            return {"accuracy": acc, "f1": f1}
        
        return compute_metrics


if __name__ == "__main__":
    from text_auto_classification.utils.task_status import TaskInfo


    with open("etc/my_configs/service_config.json") as f:
        general_conf = json.load(f)

    with open("etc/my_configs/train_config.json") as f:
        train_conf = json.load(f)


    task_info = TaskInfo()
    autoclassifier = SAAutoClassifier(general_conf, train_conf)
    autoclassifier.train_predict(task_info)

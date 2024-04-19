# SuperAnnotate Text Auto Classification #

[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)]() [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) [![CUDA 12.2](https://img.shields.io/badge/CUDA-12.2-green.svg)](https://developer.nvidia.com/cuda-12-2-0-download-archive)

This repository contains an HTTP service designed for automatic text classification for pipeline integration with the SuperAnnotate platform.

To integrate this HTTP service into your pipeline on the SuperAnnotate platform, follow these steps:

- Create and set up a text project on the SuperAnnotate platform.
- Deploy this HTTP service to a global accessible location.
- Configure a pipeline on the SuperAnnotate platform to link this service to your project.

\
<img src="pics/Main_readme_schemas.png" alt="Main schemas" width="500"/>

\
The project facilitates the automatic training of a text classification and data tagging model on the SuperAnnotate platform. \
Here's a high-level overview of the process:

1. **Annotate Data:** Annotate approximately 100 items per class using the SuperAnnotate platform.
2. **Model Fine-Tuning:** Fine-tune the text classification model using the annotated data.
3. **Prediction:** Use the fine-tuned model to predict labels for other items in your dataset.

## How it works ##

The project was created for the automatic training of a text classification and data tagging model on the SuperAnnotate platform. Everything happens in 3 main stages:

### 1. Loading and preparing data ###

- Annotations with file names are loaded from the specified project (and optional folders) from the platform.
- Document texts are also loaded through the selected integration.
- All this data is combined into a dataset and has standard processing, such as removing empty, duplicates, and extremely short/long (less than 10 and above 2000 words) texts. Texts are also preprocessed by converting them to lowercase and removing unnecessary spaces and line breaks. At any time you can change the text preprocessing function (`text_auto_classification/utils/data/data_processing.py`) to suit your needs.

### 2. Model training ###

- The training data is divided into training and validation data to evaluate the quality of the model and the learning process.
- Next, the hyperparameters are initialized, which can be customized through the training config file.
- The model's auto fine-tuning process, specified in the config, begins. All model, arguments and trainer are defined by standard HuggingFace abstractions.
- The model output layer is gonna be based on the number of classes in the training data

### 3. Prediction ###

- All downloaded data from the platform that did not yet have labels is separated during the data preparation process into a separate set for future prediction.
- At this stage, we run the texts of these elements through the model to obtain predictions.
- These predictions are then uploaded to the platform.

To configure the Pipeline and service operation from the platform side, read this [**Tutorial**](tutorial.md)

## How to run service ##

### API Service Configuration ###

You can deploy the service wherever it is convenient; one of the basic options is on a created EC2 instance. Learn about instance creation and setup [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html).

***NOTES***:

- To verify that everything is functioning correctly, try calling the healthcheck endpoint.
- Hardware requirements will depend largely on your arguments and the base model being used. However, it's recommended to utilize NVIDIA GPU architecture. For the basic configuration of the default service, it's advisable to use the following instance: [**g3s.xlarge**](https://instances.vantage.sh/aws/ec2/g3s.xlarge).
- Also, ensure that the port on which your service is deployed (8080 by default) is open to the global network. Refer to this [**tutorial**](https://stackoverflow.com/questions/5004159/opening-port-80-ec2-amazon-web-services/10454688#10454688) for guidance on opening a port on an EC2 instance.

### Pre-requirements ###

To get started with the project, you should determine all the necessary configuration files. By default, they have located in the following path: `etc/configs`. Namely, there are 3 configs:

1. **SA_config.ini**:
   - This is a configuration file for connecting work with SDK SuperAnnotate, which contains your key to the platform and is needed for authorization in SAClient. You can read more [here](https://doc.superannotate.com/docs/python-sdk#with-arguments).

2. **service_config.json**:
   - This file contains a basic field for the working of the service in general. Contains the following fields:
     - `SA_CONFIG_PATH`: The path to the first config (SA_config.ini).
     - `SA_PROJECT_NAME`, `SA_FOLDERS`: The name and optionally the folders of the project on the platform with which to work.
     - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`: AWS keys, more details [here](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html).
     - `AWS_URL_FOR_DATA_DOWNLOADS`, `AWS_URL_TO_MODEL_UPLOAD`: S3 URLs to the location of original documents and the place to save model checkpoints, respectively. More details [here](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-bucket-intro.html).

    ***NOTE***: Please ensure that the file structure on following path `AWS_URL_FOR_DATA_DOWNLOADS` matches with folders `SA_FOLDERS`

3. **train_config.json**:
   - This config contains the basic arguments necessary for training:
     - `pretrain_model`: The name of the pre-trained model with HuggingFace (it is recommended to use Bert-like model).
     - `validation_ratio`: A value from 0 to 1, representing the proportion of data that will be used to validate the model.
     - `max_length`: The maximum length of texts for the tokenizer, by default 512 is the limit for Bert-like models.
     - The remaining keys correspond to the arguments of the following `TrainingArguments` class. More details [here](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments).

After all the configs are configured as described, you can start the service. There are 2 options:

### As Python file ###

- Install Python version 3.11. More details [here](https://www.python.org/downloads/)
- Install Nvidia drivers and CUDA toolkit using, for example, this instructions: [**Nvidia drivers**](https://ubuntu.com/server/docs/nvidia-drivers-installation) and [**CUDA toolkit**](https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)
- Install dependencies: `pip install -r ./text_auto_classification/requirements.txt`
- Set the Python path variable: `export PYTHONPATH="."`
- Run the API: `uvicorn --host 0.0.0.0 --port 8080 text_auto_classification.fastapi_app:app`

### As Docker container ###

- Initialize environment variables:
  - Path to the general configuration file `DEFAULT_SERVICE_CONFIG`: `export DEFAULT_SERVICE_CONFIG=etc/configs/service_config.json`
  - Path to the configuration file with parameters for training `DEFAULT_TRAINING_CONFIG`: `export DEFAULT_TRAINING_CONFIG=etc/configs/train_config.json`
- Install Docker, Nvidia drivers, CUDA toolkit and NVIDIA Container Toolkit using, for example, this instructions: [**Docker**](https://docs.docker.com/engine/install/ubuntu/); [**Nvidia drivers**](https://ubuntu.com/server/docs/nvidia-drivers-installation); [**CUDA toolkit**](https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local); [**NVIDIA Container Toolkit**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Build the docker image: `sudo docker build -t text_auto_classification --build-arg DEFAULT_SERVICE_CONFIG=$DEFAULT_SERVICE_CONFIG --build-arg DEFAULT_TRAINING_CONFIG=$DEFAULT_TRAINING_CONFIG .`
- Run a container: `sudo docker run --gpus all -p 8080:8080 -d text_auto_classification`

## Endpoints ##

The following endpoints are available in the Text Auto Classification service:

- **GET /healthcheck**:
  - **Summary**: Ping
  - **Description**: Alive method
  - **Input Type**: None
  - **Output Type**: JSON
  - **Output Values**:
    - `{"healthy": True}`
  - **Status Codes**:
    - `200`: Successful Response

- **POST /train_predict**:
  - **Summary**: Train Predict
  - **Description**: Train model on annotated data from SA project and auto annotate other data
  - **Input Type**: None
  - **Output Type**: JSON
  - **Output Values**:
    - `{"status": "Pipeline successfully started"}`
    - `{"status": "Pipeline is already started"}`
  - **Status Codes**:
    - `200`: Pipeline successfully started
    - `429`: Pipeline is already started

- **GET /status**:
  - **Summary**: Status
  - **Description**: Method for status tracking
  - **Input Type**: None
  - **Output Type**: JSON
  - **Output Values**:
    - `{"status": "Not started"}`
    - `{"status": "Downloading data"}`
    - `{"status": "Model training"}`
    - `{"status": "Predicting other items"}`
    - `{"status": "Completed"}`
    - `{"status": "Failed"}`
  - **Status Codes**:
    - `200`: Successful Response

## Room for Improvements ##

There are several areas where the project can be further improved:

- **Implement support for multi-label classification**: Currently, the project focuses on single-label classification. Adding support for multi-label classification would enhance its versatility and applicability in various use cases.

- **Logic for working with long texts, add auto chunking**: Handling long texts efficiently is crucial for many natural language processing tasks. Implementing logic to handle long texts, such as automatic chunking, would improve the project's performance and scalability when dealing with lengthy documents.

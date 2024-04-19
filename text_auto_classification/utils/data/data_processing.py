import random

random.seed(19)


def preprocessing(data: list[dict]) -> list[dict]:
    """Preprocess data for further training.

    This function preprocesses the input data for further training by performing the following steps:
    - Dropping records with empty, short and long texts.
    - Processing text by converting it to lowercase and removing extra spaces.
    - Filtering out records with duplicate texts.

    :param data: A list containing dictionaries representing records. Each dictionary should have following keys: 'text', 'label'
    :type data: list[dict]
    :returns: The processed data in the same format.
    :rtype: list[dict]
    """
    # Drop records with empty or None texts
    data = [record for record in data if record["text"]] 
    # Drop records with short and long text. Save text with count words from 10 to 2000 
    data = [record for record in data if 10 < len(record["text"].split()) < 2000]
    
    # Process texts, lower and removing extra spaces
    data = map(lambda item: {**item, "text": item["text"].lower()}, data)
    data = map(lambda item: {**item, "text": item["text"].strip()}, data)
    data = list(map(lambda item: {**item, "text": " ".join(item["text"].split())}, data))

    data = filter_dublicates(data)

    return data


def filter_dublicates(data: list[dict]) -> list[dict]:
    """Filter duplicates in a list of records based on text and label.

    This function removes duplicates from a list of records based on the text and label fields. 
    It follows these rules:
    - If duplicates have the same text and label, only one record is kept.
    - If duplicates have the same text but different labels, all records with that text are removed.

    :param data: A list containing dictionaries representing records. Each dictionary should have following keys: 'text', 'label'
    :type data: list[dict]
    :returns: A filtered version of the input data without duplicates, in the same format.
    :rtype: list[dict]
    """
    text_dict = {}
    
    # Group records by text
    for record in data:
        text = record['text']
        if text not in text_dict:
            text_dict[text] = []
        text_dict[text].append(record)
    
    # Filter out duplicates
    filtered_data = []
    for text, records in text_dict.items():
        unique_labels = {record['label'][0] for record in records if record['label']}
        if len(unique_labels) <= 1:
            filtered_data.append(records[0]) # If all labels are the same, keep one record
    
    return filtered_data


def stratify_data_split(dataset: list[dict], test_ratio: float) -> tuple[list[dict], list[dict]]:
    """Stratify split data into two subsets while maintaining the ratio of classes in each subset.
    
    This function takes a dataset consisting of records, where each record is represented as a dictionary with keys 'text' and 'label'.
    It then splits this dataset into two subsets: a training set and a test set.
    The split is stratified, meaning that the ratio of classes in the original dataset is preserved in both subsets.

    :param dataset: A list containing dictionaries representing records. Each dictionary should have following keys: 'text', 'label'
    :type dataset: list[dict]
    :param test_ratio: The proportion of the dataset to include in the test set. Should be a float between 0 and 1.
    :type test_ratio: float
    :returns: A tuple containing two subsets of the data: the training set and the test set.
    :rtype: tuple[list[dict], list[dict]]
    """

    grouped_by_labels_data = {}

    for record in dataset:
        if record["label"] not in grouped_by_labels_data:
            grouped_by_labels_data[record["label"]] = []
        grouped_by_labels_data[record["label"]].append(record)

    train_data, test_data = [], []

    for _, data in grouped_by_labels_data.items():
        
        random.shuffle(data)
        split_index = int(len(data) * test_ratio)

        test_part, train_part = data[:split_index], data[split_index:]

        train_data.extend(train_part)
        test_data.extend(test_part)

    return train_data, test_data

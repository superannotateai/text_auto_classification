from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ModelSaver:
    """Interface for classes orieneted for save fine-tuned/trained model and tokenizer checkpoints
    """
    def save(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer) -> None:
        """Method for saving model and tokenizer

        :param model: Model for saving
        :type model: AutoModelForSequenceClassification
        :param tokenizer: Tokenizer for saving
        :type tokenizer: AutoTokenizer
        """
        raise NotImplementedError("Should have implemented this")
    
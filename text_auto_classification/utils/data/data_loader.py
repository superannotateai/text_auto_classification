class DataLoader:
    """Interface for classes orieneted for load original text of documents
    """
    def load_document(self, item_id: str) -> str:
        """Load original text of documents

        :param item_id: Name/Id of item on SA platform.
        :type item_id: str
        :return: Text of document
        :rtype: str
        """
        raise NotImplementedError("Should have implemented this")

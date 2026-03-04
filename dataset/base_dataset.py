import pandas as pd


class DatasetBase:
    """
    Stores and manages dataset records (text, annotations, and predictions).
    """

    def __init__(self, config):
        if config.records_path is None:
            self.records = pd.DataFrame(columns=['id', 'text', 'prediction',
                                                 'annotation', 'metadata', 'score', 'batch_id'])
        else:
            self.records = pd.read_csv(config.records_path)
        self.label_schema = config.label_schema

    def __len__(self):
        return len(self.records)

    def __getitem__(self, batch_idx):
        extract_records = self.records[self.records['batch_id'] == batch_idx]
        return extract_records.reset_index(drop=True)

    def update(self, records: pd.DataFrame):
        """
        Update records in dataset using 'id' as key.
        """
        if len(records) == 0:
            return

        records.set_index('id', inplace=True)
        self.records.set_index('id', inplace=True)
        self.records.update(records)

        # Remove discarded annotations
        self.records = self.records.loc[self.records["annotation"] != "Discarded"]
        self.records.reset_index(inplace=True)

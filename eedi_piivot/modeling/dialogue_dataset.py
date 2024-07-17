import json
import os
import pandas as pd
from torch.utils.data import Dataset
import datatest as dt
import ast

from eedi_piivot.utils.immutable import global_immutable

from settings import DATA_PATH

def extract_flow_message_id(row):
    meta_data = json.loads(row['example_metadata'])
    return meta_data.get('FlowGeneratorSessionInterventionMessageId', None)

def extract_flow_id(row):
    meta_data = json.loads(row['example_metadata'])
    return meta_data.get('FlowGeneratorSessionInterventionId', None)

class DialogueDataset(Dataset):
    def __init__(self, augmented_nonpii, augmented_pii, add_synthetic):
        self.__create_dataset__(augmented_nonpii, augmented_pii, add_synthetic)
        self.len = len(self.data)

    # def __init__(self, df):
    #     dt.validate(df.columns, {'FlowGeneratorSessionInterventionId', 'FlowGeneratorSessionInterventionMessageId', 'Sequence', 'Message', 'pos_labels'})
    #     self.data = df
        
    #     self.len = len(self.data)

    def __len__(self):
        return self.len
    
    def __create_dataset__(self, augmented_nonpii, augmented_pii, add_synthetic):
        # TODO change this to a path to a local file.
        if augmented_nonpii:
            dialogue_path = os.path.join(f"{DATA_PATH}/augmented-dialogue_070424.csv")
        elif add_synthetic:
            dialogue_path = os.path.join(f"{DATA_PATH}/doccano_extract_labeled_071624_w_synthetic/labeled-dialogue.csv")
        else:
            dialogue_path = os.path.join(f"{DATA_PATH}/doccano_extract_labeled_070924/labeled-dialogue.csv")

        self.data = pd.read_csv(dialogue_path)
        self.data = self.data[self.data.manually_labeled].reset_index(drop=True)
        self.data['label'] = self.data['label'].apply(ast.literal_eval)

        self.data['label'] = self.data['label'].apply(lambda x: [label for label in x if label[2] not in ['other','racial_identifier']])

        # labels = dialogue_df.label_type.unique()
        self.labels = set(label_name for labels in self.data['label'] for _, _, label_name in labels)
        self.labels.add('O')
        self.labels = sorted(self.labels)
        # TODO find a better way to do this
        global_immutable.LABELS_TO_IDS = {k: v for v, k in enumerate(self.labels)}
        global_immutable.IDS_TO_LABELS = {v: k for v, k in enumerate(self.labels)}

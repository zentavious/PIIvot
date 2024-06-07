import json
import os
import pandas as pd
from torch.utils.data import Dataset

from eedi_piivot.utils.immutable import global_immutable

from settings import DATA_PATH

def extract_flow_message_id(row):
    meta_data = json.loads(row['example_metadata'])
    return meta_data.get('FlowGeneratorSessionInterventionMessageId', None)

def extract_flow_id(row):
    meta_data = json.loads(row['example_metadata'])
    return meta_data.get('FlowGeneratorSessionInterventionId', None)

class DialogueDataset(Dataset):
    def __init__(self):
        self.__create_dataset__()
        self.len = len(self.data)

    def __len__(self):
        return self.len
    
    def __create_dataset__(self):
        dialogue_path = os.path.join(f"{DATA_PATH}/dialogue.csv")
        dialogue_df = pd.read_csv(dialogue_path)

        doccano_path = os.path.join(f"{DATA_PATH}/doccano_051024_full.csv")
        doccano_df = pd.read_csv(doccano_path)

        doccano_df['label_type'] = doccano_df['label_type'].fillna('O')
        doccano_df['flow_message_id'] = doccano_df.apply(extract_flow_message_id, axis=1)
        doccano_df['flow_id'] = doccano_df.apply(extract_flow_id, axis=1)

        doccano_df = doccano_df.drop_duplicates(subset=['flow_message_id', 'flow_id', 'start_offset', 'end_offset', 'label_type']) 

        # Remove unused label
        doccano_df = doccano_df[doccano_df['label_type'] != 'racial_identifier']

        self.labels = doccano_df.label_type.unique()
        # TODO find a better way to do this
        global_immutable.LABELS_TO_IDS = {k: v for v, k in enumerate(self.labels)}
        global_immutable.IDS_TO_LABELS = {v: k for v, k in enumerate(self.labels)}

        flow_message_ids = set(doccano_df['flow_message_id'])
        self.data = dialogue_df[dialogue_df['FlowGeneratorSessionInterventionMessageId'].isin(flow_message_ids)][['FlowGeneratorSessionInterventionId', 'FlowGeneratorSessionInterventionMessageId', 'Sequence', 'Message']].reset_index(drop=True)

        self.data['pos_labels'] = [[] for _ in range(len(self.data))]

        for index, row in self.data.iterrows():
            corresponding_rows = doccano_df[(doccano_df['flow_message_id'] == row['FlowGeneratorSessionInterventionMessageId'])&(doccano_df['label_type'] != 'O')]
            
            gt_labels_list = [(start_offset, end_offset, label_type) for start_offset, end_offset, label_type 
                            in zip(corresponding_rows['start_offset'], corresponding_rows['end_offset'], corresponding_rows['label_type'])]
            
            self.data.at[index, 'pos_labels'] = gt_labels_list

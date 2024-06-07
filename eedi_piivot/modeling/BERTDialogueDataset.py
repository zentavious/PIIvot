import json
import os
import pandas as pd
import numpy as np
import torch
from collections import Counter

from eedi_piivot.utils.immutable import global_immutable
from .dialogue_dataset import DialogueDataset

def extract_flow_message_id(row):
    meta_data = json.loads(row['example_metadata'])
    return meta_data.get('FlowGeneratorSessionInterventionMessageId', None)

def extract_flow_id(row):
    meta_data = json.loads(row['example_metadata'])
    return meta_data.get('FlowGeneratorSessionInterventionId', None)

# TODO fix dataset to add encodings + propgate labels at __getitem__??? Refer to tutorial
class BERTDialogueDataset(DialogueDataset):
    def __init__(self, max_length, tokenizer):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = max_length
        self.encode_dataframe()
        self.assign_minority_labels()

    def assign_minority_labels(self):
        all_labels = [label for sublist in self.data['encoded_labels'] for label in sublist]
        label_counts = Counter(all_labels)
        
        ordered_labels = [label for label, count in sorted(label_counts.items(), key=lambda item: item[1])]

        self.data['minority_label'] = self.data.apply(lambda row: self.assign_minority_label(row['encoded_labels'], ordered_labels), axis=1)

    def assign_minority_label(self, encoded_labels, ordered_labels):
        for label in ordered_labels:
            if label in encoded_labels:
                return label
            
    def assign_labels_bert(self, encoding, pos_labels):
        
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] != mapping[1]:
                encoded_labels[idx] = global_immutable.LABELS_TO_IDS["O"] # default to out

                for start_index, end_index, label_type in pos_labels:
                    if start_index <= mapping[0] < end_index or start_index < mapping[1] <= end_index:
                        encoded_labels[idx] = global_immutable.LABELS_TO_IDS[label_type]
                        break
                
        return encoded_labels

    def encode_message(self, message):
        encoding = self.tokenizer(message,
                            return_offsets_mapping=True, 
                            padding='max_length', 
                            truncation=True, 
                            max_length=self.max_len)
        return encoding

    def tokenize_message(self, encoding):
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        return tokens
    
    def encode_dataframe(self):
        self.data['bert_encoding'] = self.data.apply(lambda row: self.encode_message(row['Message']), axis=1)
        self.data['encoded_labels'] = self.data.apply(lambda row: self.assign_labels_bert(row['bert_encoding'], row['pos_labels']), axis=1)
        self.data['tokens'] = self.data.apply(lambda row: self.tokenize_message(row['bert_encoding']), axis=1)

    def __getitem__(self, index):
        # step 1: get the encoding and labels 
        encoding = self.data.bert_encoding[index] 
        labels = self.data.encoded_labels[index] 

        # step 2: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(labels)
        
        return item

    def __len__(self):
        return self.len
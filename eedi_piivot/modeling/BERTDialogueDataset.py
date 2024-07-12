import json
import os
from typing import List
import pandas as pd
import numpy as np
import torch
from collections import Counter
import random

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
    def __init__(self, max_length, tokenizer, augmented_nonpii, augmented_pii):
        super().__init__(augmented_nonpii, augmented_pii)

        self.tokenizer = tokenizer
        self.max_len = max_length
        self.encode_dataframe()
        self.assign_minority_labels()

    # def __init__(self, df, max_length, tokenizer):
    #     super().__init__(df)

    #     self.tokenizer = tokenizer
    #     self.max_len = max_length
    #     self.encode_dataframe()
    #     self.assign_minority_labels()

    def get_non_augmented_df(self):
        if 'is_augmented' in self.data.columns:
            return self.data[self.data['is_augmented'] == False]
        else:
            return self.data
    
    
    def get_df_from_indicies(self, indices: List[int]) -> List[int]:
        return self.data.loc[indices]
        

    def get_non_augmented_indices(self):
        if 'is_augmented' in self.data.columns:
            return self.data.index[self.data['is_augmented'] == False].tolist()
        else:
            return self.data.index.tolist()
            
    def expand_group_indices(self, indices: List[int]) -> List[int]:
        unique_ids = self.data.loc[indices, 'FlowGeneratorSessionInterventionMessageId'].unique()

        # Find all indices in the DataFrame that match these FlowGeneratorSessionInterventionMessageId values
        exapanded_indices = self.data.index[self.data['FlowGeneratorSessionInterventionMessageId'].isin(unique_ids)].tolist()
        random.shuffle(exapanded_indices)
        return exapanded_indices


    def assign_minority_labels(self):
        all_labels = [label for sublist in self.data['encoded_labels'] for label in sublist]
        label_counts = Counter(all_labels)
        
        ordered_labels = [label for label, count in sorted(label_counts.items(), key=lambda item: item[1])]

        self.data['minority_label'] = self.data.apply(lambda row: self.assign_minority_label(row['encoded_labels'], ordered_labels), axis=1)

        minority_label_counts = Counter(self.get_non_augmented_df()['minority_label'].tolist())
        below_min_labels = [key for key, count in minority_label_counts.items() if count < 3]
        if below_min_labels:
            print(f"Removing labels with less than 3 supporting messages: {[f"{global_immutable.IDS_TO_LABELS[label]}:{label}" for label in below_min_labels]}.")
            self.data['encoded_labels'] = self.data['encoded_labels'].apply(lambda x: [label if label not in below_min_labels else global_immutable.LABELS_TO_IDS['O'] for label in x ])
            
            # if we had to remove labels, we need to re-assign minority labels.
            self.assign_minority_labels()

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
        temp = len(encoding.data['input_ids'])

        if len(encoding.data['input_ids']) > self.max_len: 
            print(f"Message truncated : {message}")

        return encoding

    def tokenize_message(self, encoding):
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        return tokens
    
    def encode_dataframe(self):
        self.data['bert_encoding'] = self.data.apply(lambda row: self.encode_message(row['Message']), axis=1)
        self.data['encoded_labels'] = self.data.apply(lambda row: self.assign_labels_bert(row['bert_encoding'], row['label']), axis=1)
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


class MultiSentenceBERTDialogueDataset(BERTDialogueDataset):
    def __init__(self, max_length, tokenizer, augmented_nonpii, augmented_pii):
        super().__init__(max_length, tokenizer, augmented_nonpii, augmented_pii)

    # def __init__(self, df, max_length, tokenizer):
    #     super().__init__(df, max_length, tokenizer)

    def encode_message(self, message, previous_message, next_message):
        combined_message = f"{previous_message} {message} {next_message}".strip()
        
        encoding = self.tokenizer(
            combined_message,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=False,
            max_length=self.max_len
        )
        temp = len(encoding.data['input_ids'])
        
        if len(encoding.data['input_ids']) > self.max_len: 
            print(f"Message truncated : {message}")

        # prev_tokens = self.tokenizer.tokenize(previous_message)

        # encoding['previous_offset'] = len(prev_tokens) + 1 if previous_message else 0
        encoding['window_start'] = len(previous_message) if previous_message else 0 # not including +1 will include the proceeding space.
        encoding['window_end'] = len(previous_message) + 1 + len(message) if previous_message else len(message)
        return encoding

    def assign_labels_bert(self, encoding, pos_labels):
        
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] != mapping[1] and mapping[0] >= encoding['window_start'] and mapping[1] <= encoding['window_end']:
                encoded_labels[idx] = global_immutable.LABELS_TO_IDS["O"] # default to out

                for start_index, end_index, label_type in pos_labels:
                    if start_index + encoding['window_start'] <= mapping[0] < end_index + encoding['window_start'] + 1 or start_index + encoding['window_start'] < mapping[1] <= end_index + encoding['window_start'] + 1:
                        encoded_labels[idx] = global_immutable.LABELS_TO_IDS[label_type]
                        break
                
        return encoded_labels
    
    def encode_dataframe(self):
        messages = self.data['Message'].tolist()
        bert_encodings = []
        
        for i, message in enumerate(messages):
            current_id = self.data.iloc[i]['FlowGeneratorSessionInterventionMessageId']
            current_session_id = self.data.iloc[i]['FlowGeneratorSessionInterventionId']
            current_is_augmented = self.data.iloc[i]['is_augmented']
            
            # Filter for previous and next messages
            previous_messages = self.data[
                (self.data['FlowGeneratorSessionInterventionId'] == current_session_id) &
                (self.data['FlowGeneratorSessionInterventionMessageId'] == current_id - 1) &
                (self.data['is_augmented'] == current_is_augmented)
            ]
            
            next_messages = self.data[
                (self.data['FlowGeneratorSessionInterventionId'] == current_session_id) &
                (self.data['FlowGeneratorSessionInterventionMessageId'] == current_id + 1) &
                (self.data['is_augmented'] == current_is_augmented)
            ]
            
            # Randomly select previous and next messages if available
            previous_message = random.choice(previous_messages['Message'].tolist()) if not previous_messages.empty else ""
            next_message = random.choice(next_messages['Message'].tolist()) if not next_messages.empty else ""
            
            # Encode the message with its context
            bert_encoding = self.encode_message(message, previous_message, next_message)
            bert_encodings.append(bert_encoding)
        
        self.data['bert_encoding'] = bert_encodings
        self.data['encoded_labels'] = self.data.apply(lambda row: self.assign_labels_bert(row['bert_encoding'], row['pos_labels']), axis=1)
        self.data['tokens'] = self.data.apply(lambda row: self.tokenize_message(row['bert_encoding']), axis=1)

    def encode_dataframe_old(self):
        messages = self.data['Message'].tolist()
        bert_encodings = []

        last_dialogue_id = -1
        for i, message in enumerate(messages):
            dialogue_id = self.data.iloc[i]['FlowGeneratorSessionInterventionId']
            previous_message = messages[i - 1] if i > 0 and dialogue_id == last_dialogue_id else ""
            next_message = messages[i + 1] if i < len(messages) - 1 and dialogue_id == self.data.iloc[i + 1]['FlowGeneratorSessionInterventionId'] else ""
            bert_encoding = self.encode_message(message, previous_message, next_message)
            bert_encodings.append(bert_encoding)
            last_dialogue_id = dialogue_id

        self.data['bert_encoding'] = bert_encodings

        
        self.data['encoded_labels'] = self.data.apply(lambda row: self.assign_labels_bert(row['bert_encoding'], row['pos_labels']), axis=1)
        self.data['tokens'] = self.data.apply(lambda row: self.tokenize_message(row['bert_encoding']), axis=1) #TODO off by one error on the mapping

    # def __getitem__(self, index):
    #     # step 1: get the encoding and labels 
    #     encoding = self.data.bert_encoding[index] 
    #     labels = self.data.encoded_labels[index] 

    #     # step 2: turn everything into PyTorch tensors
    #     item = {key: torch.as_tensor(val) for key, val in encoding.items()}
    #     item['labels'] = torch.as_tensor(labels)
        
    #     return item
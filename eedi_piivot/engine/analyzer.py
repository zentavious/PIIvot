import json
from pydantic import ValidationError
import torch
from transformers import pipeline
import pandas as pd
from typing import List

from eedi_piivot.utils.helpers import initialize_model_and_optimizer
from eedi_piivot.experiments.config import AnayzerConfig
from eedi_piivot.utils.tokenizer_factory import (
    create_tokenizer,
)

DEFAULT_DEVICE = 'cpu'

class Analyzer():
    '''Analyzer Engine.'''
    
    def __init__(self, config_path=None, device=DEFAULT_DEVICE):
        
        with open(config_path, "r") as f:
            raw_data = json.load(f)

        try:
            self.config = AnayzerConfig(**raw_data)
        except ValidationError as e:
            print(e)

        
        self.model, _ = initialize_model_and_optimizer(self.config.model,
                                                       self.config.optimizer,
                                                       device)
        
        self.tokenizer = create_tokenizer(self.config.model.params.name, 
                                          self.config.model.params.from_pretrained, 
                                          self.config.model.pretrained_params.pretrained_model_name_or_path)
        

        checkpoint = torch.load(self.config.checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.id_to_label = checkpoint['id_to_label']
 
        self.token_classifier = pipeline(task="ner", model=self.model, tokenizer=self.tokenizer)
    
    def classify_message(self, message: str) -> List:
        labels = self.token_classifier(message)
        last_span = None
        word_labels = []
        for label in labels:
            label_name = self.id_to_label[int(label['entity'][6:])]

            if label_name != 'O':
                if last_span is not None and label_name == last_span[2]:
                    last_span = tuple([last_span[0], label['end'], last_span[2]])
                    word_labels[-1] = last_span
                else:
                    last_span = tuple([label['start'], label['end'], label_name])
                    word_labels.append(last_span)
            else:
                last_span = None
        return word_labels

        
    def analyze(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame: #TODO batch inputs to utilize GPU

        labels_df = pd.DataFrame(index=df.index)

        for column in columns:
            if column in df.columns:
                labels_df[f"{column}_labels"] = df[column].apply(self.classify_message) #{'entity': 'LABEL_1', 'score': 0.99995375, 'index': 1, 'word': '▁Hi', 'start': 0, 'end': 2}
            else:
                raise ValueError(f"Column '{column}' does not exist in the input dataframe")

        return labels_df

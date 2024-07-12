import json
from pydantic import ValidationError
from eedi_piivot.utils.config import AnonymizerConfig
import pandas as pd
from typing import List

from eedi_piivot.engine.analyzer import Analyzer

class Anonymizer():
    '''Anonymizer Engine.'''
    
    def __init__(self, config_path, gpt_client):
        with open(config_path, "r") as f:
            raw_data = json.load(f)

        try:
            self.config = AnonymizerConfig(**raw_data)
        except ValidationError as e:
            print(e)
        self.labeledAnonymizer = LabeledAnonymizer()

    def anonymize(self, df: pd.DataFrame, columns: List[str], labels: List[str], column_replacement_functions: List = None) -> pd.DataFrame:
        
        anonymized_df = pd.DataFrame(index=df.index)

        for column, label in zip(columns, labels):
            if column in df.columns:
                anonymized_df[[column, label]] = df.apply(lambda row: self.labeledAnonymizer.anonymize(row[column], row[label]), axis=1, result_type ='expand')
            else:
                raise ValueError(f"Column '{column}' does not exist in the input dataframe")

        return anonymized_df
    
    
    # def anonymize(self, analyzer: Analyzer, df: pd.DataFrame, columns: List[str], column_replacement_functions: List = None) -> pd.DataFrame:
    #     labels_df = analyzer.analyze(df, columns)
    #     df = pd.concat([df, labels_df], axis=1)
    #     return self.anonymize(df, columns, labels_df.columns, column_replacement_functions)


# class NameAnonymizer(GenericAnonymizer):

# class DateAnonymizer(Genericnonymizer):

# class LocationAnonymizer(Genericnonymizer):

class LabeledAnonymizer():
    def anonymize(self, message: str, labels: List[tuple[int, int, str]]):
        
        # sort labels to ensure consistent indexing
        labels = sorted(labels, key=lambda x: x[1])

        for i in range(len(labels)):
            label = labels[i]
            message = message[:label[0]] + '[[' + label[2] + ']]' + message[label[1]:]
            offset = len(f"[[{label[2]}]]")
            labels[i] = (label[0], label[0] + offset, label[2])
            
            # update down index label indicies by the new length of the 
            for j in range(i + 1,len(labels)):
                labels[j] = (labels[j][0] + offset, labels[j][1] + offset, labels[j][2])

        return message, labels


class GenericAnonymizer():
    def __init__(self, gpt_client):
        self.gpt_client = gpt_client
        self.prompt = ""
        self.temperature=0.2
        self.max_tokens=512
        self.frequency_penalty=0.0
    
    def anonymize(self, message: str, labels: List[tuple[int, int, str]]):


        chat_completion = client.chat.completions.create(
            messages = message,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            frequency_penalty=self.frequency_penalty,
            model="gpt-3.5-turbo",
        )


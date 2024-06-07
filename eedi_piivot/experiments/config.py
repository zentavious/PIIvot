"""A module for getting and setting experiment configs."""

import datetime
from typing import Literal, Optional

from pydantic import BaseModel, PositiveFloat, PositiveInt, confloat, validator

class BatchParamsConfig(BaseModel):
    batch_size: PositiveInt
    shuffle: bool
    num_workers: int

class InputDataConfig(BaseModel):
    path: str
    header: bool
    processed_date: datetime.date
    split: bool
    k_folds: Optional[PositiveInt]
    train_split: confloat(ge=0.0, le=1.0)  # type: ignore
    train_params: BatchParamsConfig
    valid_split: Optional[confloat(ge=0.0, le=1.0)]  # type: ignore
    valid_params: BatchParamsConfig

    @validator('valid_split')
    def ensure_valid_split(cls, v, values, **kwargs):
        if v is None and values['k_folds'] is None:
            raise KeyError('k_folds or valid_split are required')

class ModelParamsConfig(BaseModel):
    model_name: Literal["BERT", "DeBERTa"]
    from_pretrained: bool
    max_len: PositiveInt


class PretrainedModelParamsConfig(BaseModel):
    pretrained_model_name_or_path: Literal["bert-base-cased", "microsoft/deberta-v3-base"]
    num_labels: PositiveInt

class ModelConfig(BaseModel):
    model_params: ModelParamsConfig
    pretrained_params: PretrainedModelParamsConfig

class OptimizerParamsConfig(BaseModel):
    lr: PositiveFloat

class OptimizerConfig(BaseModel):
    name: Literal["Adam"]
    params: OptimizerParamsConfig

class TrainerConfig(BaseModel):
    name: Literal["DialogueTrainer"]
    val_every: PositiveInt
    epochs: int
    use_tqdm: bool
    grad_clipping_max_norm: PositiveInt
    optimizer: OptimizerConfig
    resume_checkpoint_path: Optional[str]

class ExperimentConfig(BaseModel):
    model: ModelConfig
    trainer: TrainerConfig
    seed: PositiveInt

class Config(BaseModel):
    input_data: InputDataConfig
    experiment: ExperimentConfig
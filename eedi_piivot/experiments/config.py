"""A module for getting and setting experiment configs."""

import datetime
from typing import Literal, Optional

from pydantic import BaseModel, PositiveFloat, PositiveInt, confloat

class BatchParamsConfig(BaseModel):
    batch_size: PositiveInt
    shuffle: bool
    num_workers: int

class InputDataConfig(BaseModel):
    path: str
    header: bool
    processed_date: datetime.date
    split: bool
    max_len: PositiveInt
    train_split: confloat(ge=0.0, le=1.0)  # type: ignore
    train_params: BatchParamsConfig
    valid_split: confloat(ge=0.0, le=1.0)  # type: ignore
    valid_params: BatchParamsConfig
    # dataset_type: DatasetTypeConfig
    # sampler: SamplerConfig
    # batch_size: PositiveInt

class ModelParamsConfig(BaseModel):
    model_name: Literal["BERT"]
    from_pretrained: bool


class PretrainedModelParamsConfig(BaseModel):
    pretrained_model_name_or_path: Literal["bert-base-cased"]
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
    epochs: PositiveInt
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
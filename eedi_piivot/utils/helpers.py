import torch
from typing import Tuple

from eedi_piivot.utils.console import console
from .model_factory import create_model
from .optimizer_factory import create_optimizer

def initialize_model_and_optimizer(model_config, optimizer_config, device) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    console.rule(
        f"Initializing the {model_config.params.name} model."
    )
    
    model = create_model(model_config.params.name, 
                         model_config.params.from_pretrained, 
                         **model_config.pretrained_params.model_dump())
    
    model.to(device)

    optimizer = create_optimizer(optimizer_config.name,
                                 model.parameters(),
                                 **optimizer_config.params.model_dump())
    
    return (model, optimizer)
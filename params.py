from typing import Tuple, List
import torch
from original_code.synflow.Utils import load
from original_code.synflow.Pruners.pruners import Pruner
from datetime import datetime
from pathlib import Path


class State:
    prune_loader = None
    train_loader = None
    test_loader = None
    model: torch.nn.Module
    loss: torch.nn.modules.loss._Loss
    scheduler: torch.optim.lr_scheduler.MultiStepLR 
    optimizer = torch.optim.Optimizer 
    pruner: Pruner

    def save(self, path:str = None):
        if not path:
            path = "./saves/" + str(datetime.now()).replace(" ", "_").replace(":", "-").replace(".","_")
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model, path + "/model.pt")
        torch.save(self.optimizer, path + "/optimizer.pt")
        torch.save(self.scheduler, path + "/scheduler.pt")

    def load(self, path:str):
        self.model = torch.load(path + "/model.pt")
        self.optimizer = torch.load(path + "/optimizer.pt")
        self.scheduler = torch.load(path + "/scheduler.pt")

class ModelParams:
    model: str
    model_class: str
    lr: float
    lr_drop_rate: float 
    lr_drops: List
    weight_decay: float
    dense_classifier: bool
    pretrained: bool
    optimizer: str
 
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

class DataParams:
    dataset: str = "mnist"
    num_classes: int
    input_shape: Tuple[int, int, int]
    train_batch_size: int
    test_batch_size: int
    prune_batch_size: int
    workers: int
    prune_dataset_ratio: int

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
        self.input_shape, self.num_classes = load.dimension(self.dataset) 

 # TODO: cleanup?
class PruningParams:
    strategy: str
    sparsity: float
    prune_epochs: int
    prune_bias: bool
    prune_batchnorm: bool
    prune_residual: bool
    compression_schedule: str
    mask_scope: str
    reinitialize: bool
    prune_train_mode: bool
    shuffle: bool
    invert: bool

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)



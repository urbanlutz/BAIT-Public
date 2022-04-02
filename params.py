from typing import Tuple, List
import torch
from original_code.synflow.Utils import load


class State:
    prune_loader = None
    train_loader = None
    test_loader = None
    model: torch.nn.Module
    loss = None
    scheduler = None  
    optimizer = None  



class ModelParams:
    model: str
    model_class: str
    lr: float = 0.001
    lr_drop_rate: float = 0.1
    lr_drops: List = []
    weight_decay: float = 0.0
    optimizer: str
    dense_classifier: bool = False
    pretrained: bool = False
    optimizer: str = "adam"
 
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

class DataParams:
    dataset: str = "mnist"
    num_classes: int
    input_shape: Tuple[int, int, int]
    train_batch_size: int = 64
    test_batch_size: int = 256
    prune_batch_size: int = 256
    workers: int = 4
    prune_dataset_ratio: int = 10

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
        self.input_shape, self.num_classes = load.dimension(self.dataset) 

class PruningParams:
    strategy: str = "synflow"
    sparsity: float = 0.5
    prune_epochs: int = 1
    prune_bias: bool = False
    prune_batchnorm: bool = False
    prune_residual: bool = False
    compression_schedule: str = "exponential"
    mask_scope: str = "global"
    reinitialize: bool = False
    prune_train_mode: bool = False
    shuffle: bool = False
    invert: bool = False

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)



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
 
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

class DataParams:
    dataset: str
    num_classes: int
    input_shape: Tuple[int, int, int]
    train_batch_size: int
    test_batch_size: int
    prune_batch_size: int
    workers: int
    prune_dataset_ratio: float

    def __init__(self, dataset="mnist", train_batch_size=64, test_batch_size=256, prune_batch_size=256, workers=4):
        self.dataset = dataset
        self.input_shape, self.num_classes = load.dimension(dataset) 
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.prune_batch_size = prune_batch_size
        self.workers = workers

class PruningParams:
    strategy: str
    sparsity: float
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



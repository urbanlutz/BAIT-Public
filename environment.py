import torch
from original_code.synflow.Utils import load

class Environment:
    device: torch.device
    verbose: bool
    seed: int

    def __init__(self, verbose = False, seed=None):
        self.verbose = verbose
        self.device = load.device(0)

        if seed:
            self.seed = seed
            torch.manual_seed(seed)

ENV = Environment()
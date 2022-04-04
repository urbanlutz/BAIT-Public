from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Monitor:
    writer = None

    def __init__(self, experiment:str = None):
        self.iteration = 1

        if experiment:
            name = f"{experiment}_{datetime.now().split(' ')[0]}"
        else:
            name = None

        self.writer = SummaryWriter(name)

    def __del__(self):
        self.writer.flush()

    def inc(self):
        self.iteration += 1

    def track(self, name: str, value:float):
        self.writer.add_scalar(name, value, self.iteration)

    def track_param(self, param, value):
        if not isinstance(value, (int, float, str, bool)):
            value = str(value)
        self.writer.add_hparams({param:value}, {})

MONITOR = Monitor()
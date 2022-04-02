from torch.utils.tensorboard import SummaryWriter

class Monitor:
    writer = SummaryWriter()
    iteration = 1

    def __del__(self):
        self.writer.flush()

    def reset(self):
        self.writer = SummaryWriter()
        self.iteration = 1

    def inc(self):
        self.iteration += 1

    def track(self, name: str, value:float):
        self.writer.add_scalar(name, value, self.iteration)

    def track_param(self, param, value):
        if not isinstance(value, (int, float, str, bool)):
            value = str(value)
        self.writer.add_hparams({param:value}, {})

MONITOR = Monitor()
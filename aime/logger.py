import json
import logging
import os
from pprint import pformat

from torch.utils.tensorboard import SummaryWriter


def get_default_logger(root: str):
    logger = ListLogger(root)
    logger.add(TerminalLogger)
    logger.add(TensorboardLogger)
    logger.add(JsonlLogger)
    return logger


class Logger:
    def __init__(self, root: str) -> None:
        self.root = root

    def __call__(self, metrics: dict, step: int):
        raise NotImplementedError


class ListLogger(Logger):
    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.loggers = []

    def add(self, logger_class):
        self.loggers.append(logger_class(self.root))

    def __call__(self, metrics: dict, step: int):
        for logger in self.loggers:
            logger(metrics, step)


class TerminalLogger(Logger):
    """log metrics to the terminal"""

    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.log = logging.getLogger("log")

    def __call__(self, metrics: dict, step: int):
        metrics = metrics.copy()
        for k in list(metrics.keys()):
            if "video" in k:
                metrics.pop(k)
        self.log.info(pformat(metrics))


class TensorboardLogger(Logger):
    """log metrics to the tensorboard"""

    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.writer = SummaryWriter(self.root)

    def __call__(self, metrics: dict, step: int):
        for k, v in metrics.items():
            if "video" in k:
                v = v.permute(0, 3, 1, 2).unsqueeze(dim=0)
                v = v / 255  # hard coded covert
                self.writer.add_video(k, v, global_step=step, fps=25)
            elif isinstance(v, list):
                continue
            else:
                self.writer.add_scalar(k, v, global_step=step)


class JsonlLogger(Logger):
    """log metrics to a jsonl file"""

    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.writer = open(os.path.join(self.root, "metrics.jsonl"), "w")

    def __call__(self, metrics: dict, step: int):
        metrics = metrics.copy()
        metrics["step"] = step
        for k in list(metrics.keys()):
            if "video" in k:
                metrics.pop(k)
        self.writer.write(json.dumps(metrics) + "\n")
        self.writer.flush()

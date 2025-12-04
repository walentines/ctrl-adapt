from __future__ import annotations

from collections import defaultdict

import numpy as np

import time
from .tensorboard_logger import TensorboardLogger
from .state import EvaluatorState, StoreKey, TrainerState


class TimeLogger(TensorboardLogger):
    def __init__(self, **kwargs):
        """
        Callback for computing and logging a metric.
        """
        super().__init__(**kwargs)
        self.start_time_train = 0
        self.start_time_val = 0
        self.end_time_train = 0
        self.end_time_val = 0

    def on_training_epoch_start(self):
        self.start_time_train = time.time()

    
    def on_training_epoch_end(self):
        self.end_time_train = time.time()
        delta_time = self.end_time_train - self.start_time_train
        self.log_scalar(f'epoch/train/time', delta_time, log_to_console=False)

    def on_validation_epoch_start(self):
        self.start_time_val = time.time()

    
    def on_validation_epoch_end(self):
        self.end_time_train = time.time()
        delta_time = self.end_time_train - self.start_time_val
        self.log_scalar(f'epoch/validation/time', delta_time, log_to_console=False)

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .tensorboard_logger import TensorboardLogger
from .state import EvaluatorState, StoreKey, TrainerState


class MetricLogger(TensorboardLogger):
    def __init__(self, metric_name,**kwargs):
        """
        Callback for computing and logging a metric.
        """
        super().__init__(**kwargs)
        self.metric_name = metric_name
        # self.train_epoch_losses = defaultdict(list)
        # self.eval_epoch_losses = defaultdict(list)
    
    def on_metric_calculation(self, value):
        self.log_scalar(f'epoch/{self.metric_name}', value, log_to_console=False)

    # def on_train_batch_end(self, name, value) -> None:
    #     self.log_scalar(f'batch/train/{name}', value, log_to_console=False)
    #     self.train_epoch_losses[name].append(value)

    # def on_evaluation_batch_end(self, name, value) -> None:
    #     self.log_scalar(f'batch/eval/{name}', value, log_to_console=False)
    #     self.eval_epoch_losses[name].append(value)

    # def on_evaluation_end(self) -> None:

    #     if len(self.train_epoch_losses) > 0 and len(self.eval_epoch_losses) > 0:
    #         # log both eval and train mean losses over the epoch
    #         for name, train_values in self.train_epoch_losses.items():
    #             mean_train = np.mean(train_values)
    #             mean_eval = np.mean(self.eval_epoch_losses[name])

    #             self.log_scalars(f'epoch/{name}', {
    #                     'train': mean_train,
    #                     'eval': mean_eval,
    #                 }, log_to_console=True
    #             )
    #     else:
    #         # log either train or eval mean loss over the epoch
    #         losses_dict = {}
    #         if len(self.train_epoch_losses) > 0:
    #             losses_dict = self.train_epoch_losses
    #             stage = 'train'
    #         elif len(self.eval_epoch_losses) > 0:
    #             losses_dict = self.eval_epoch_losses
    #             stage = 'eval'

    #         for name, values in losses_dict.items():
    #             self.log_scalar(f'epoch/{stage}/{name}', float(np.mean(values)))

    #     # reset history
    #     self.train_epoch_losses = defaultdict(list)
    #     self.eval_epoch_losses = defaultdict(list)
    #     self.lrs = []
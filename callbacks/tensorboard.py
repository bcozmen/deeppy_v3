"""TensorBoard training diagnostics."""

from contextlib import contextmanager

import torch
from torch.utils.tensorboard import SummaryWriter

from .base import Callback


class TensorBoardLogger(Callback):
    """Log losses, learning rates, and optional gradients to TensorBoard."""

    def __init__(self, log_dir="runs", writer=None, log_gradients=False):
        self.log_dir = log_dir
        self.writer = writer
        self.log_gradients = log_gradients
        self.runtime = None
        self.phase = "train"
        self.step = 0

    def bind(self, runtime):
        """Bind the logger to a Runtime for rank-aware logging."""
        self.runtime = runtime
        return self

    @contextmanager
    def scope(self, phase, step):
        """Set the default phase and step while model code is executing."""
        previous = self.phase, self.step
        self.phase, self.step = phase, step
        try:
            yield self
        finally:
            self.phase, self.step = previous

    def _ensure_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)
        return self.writer

    def _enabled(self):
        return self.runtime is None or self.runtime.is_main_process

    def log(self, name, value, *, step=None, phase=None):
        """Log a scalar or tensor metric from model code."""
        if not self._enabled():
            return
        value = value.detach() if torch.is_tensor(value) else value
        step = self.step if step is None else step
        tag = f"{phase or self.phase}/{name}"
        writer = self._ensure_writer()
        if torch.is_tensor(value) and value.numel() > 1:
            writer.add_histogram(tag, value.float().cpu(), step)
        else:
            scalar = value.item() if torch.is_tensor(value) else value
            writer.add_scalar(tag, scalar, step)

    def log_histogram(self, name, value, *, step=None, phase=None):
        """Log a tensor distribution from model code."""
        if not self._enabled():
            return
        step = self.step if step is None else step
        self._ensure_writer().add_histogram(
            f"{phase or self.phase}/{name}", value.detach().float().cpu(), step
        )

    def on_optimizer_step_end(self, trainer, updated):
        if not updated or not trainer.runtime.is_main_process:
            return

        step = trainer.global_step
        if trainer.latest_loss is not None:
            self.log("loss", trainer.latest_loss, step=step, phase="train")

        for index, optimizer in enumerate(trainer.optimizers):
            for group_index, group in enumerate(optimizer.raw_optimizer.param_groups):
                self.log(
                    f"optimizer/{index}/lr/{group_index}",
                    group["lr"],
                    step=step,
                    phase="train",
                )

        if self.log_gradients:
            for name, parameter in trainer.named_parameters():
                if parameter.grad is not None:
                    self.log(
                        f"gradients/{name}",
                        parameter.grad.detach().norm(),
                        step=step,
                        phase="train",
                    )

    def close(self):
        if self.writer is not None:
            self.writer.close()


# Kept as an alias for code written against the initial callback API. New code
# should use TensorBoardLogger; XAI is reserved for explainability callbacks.
XAI = TensorBoardLogger
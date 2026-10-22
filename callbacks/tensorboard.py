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

    def _log_optimizer_diagnostics(self, trainer, *, include_gradients=False):
        """Log parameters, gradients, hyperparameters, and optimizer state."""
        for optimizer_index, optimizer in enumerate(trainer.optimizers):
            raw_optimizer = optimizer.raw_optimizer
            parameters = {
                id(parameter): (name, parameter)
                for name, parameter in trainer.named_parameters()
            }

            for group_index, group in enumerate(raw_optimizer.param_groups):
                prefix = f"optimizer/{optimizer_index}/group/{group_index}"
                for name, value in group.items():
                    if name == "params":
                        continue
                    if isinstance(value, (int, float)):
                        self.log(f"{prefix}/{name}", value, phase="train")

                for parameter in group["params"]:
                    parameter_name, parameter = parameters.get(
                        id(parameter), (str(id(parameter)), parameter)
                    )
                    parameter_prefix = (
                        f"optimizer/{optimizer_index}/parameters/{parameter_name}"
                    )
                    self.log(
                        f"{parameter_prefix}/norm",
                        parameter.detach().norm(),
                        phase="train",
                    )
                    if include_gradients and parameter.grad is not None:
                        self.log(
                            f"{parameter_prefix}/gradient_norm",
                            parameter.grad.detach().norm(),
                            phase="train",
                        )

                    state = raw_optimizer.state.get(parameter, {})
                    for state_name, state_value in state.items():
                        state_prefix = f"{parameter_prefix}/state/{state_name}"
                        if torch.is_tensor(state_value):
                            if state_value.numel() == 1:
                                self.log(state_prefix, state_value, phase="train")
                            else:
                                self.log(
                                    f"{state_prefix}/norm",
                                    state_value.norm(),
                                    phase="train",
                                )
                                self.log_histogram(
                                    state_prefix, state_value, phase="train"
                                )
                        elif isinstance(state_value, (int, float)):
                            self.log(state_prefix, state_value, phase="train")

    def on_before_optimizer_step(self, trainer):
        """Capture gradients before the optimizer clears them."""
        if self.log_gradients and trainer.runtime.is_main_process:
            with self.scope("train", trainer.global_step):
                self._log_optimizer_diagnostics(trainer, include_gradients=True)

    def on_optimizer_step_end(self, trainer, updated):
        if not updated or not trainer.runtime.is_main_process:
            return

        step = trainer.global_step
        if trainer.latest_loss is not None:
            self.log("loss", trainer.latest_loss, step=step, phase="train")

        with self.scope("train", step):
            self._log_optimizer_diagnostics(trainer, include_gradients=False)

        for index, optimizer in enumerate(trainer.optimizers):
            for group_index, group in enumerate(optimizer.raw_optimizer.param_groups):
                self.log(
                    f"optimizer/{index}/lr/{group_index}",
                    group["lr"],
                    step=step,
                    phase="train",
                )

    def close(self):
        if self.writer is not None:
            self.writer.close()


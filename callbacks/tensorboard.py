"""TensorBoard training diagnostics."""

from torch.utils.tensorboard import SummaryWriter

from .base import Callback


class TensorBoardLogger(Callback):
    """Log losses, learning rates, and optional gradients to TensorBoard."""

    def __init__(self, log_dir="runs", writer=None, log_gradients=False):
        self.log_dir = log_dir
        self.writer = writer
        self.log_gradients = log_gradients

    def on_optimizer_step_end(self, trainer, updated):
        if not updated or not trainer.runtime.is_main_process:
            return

        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)

        step = trainer.global_step
        if trainer.latest_loss is not None:
            self.writer.add_scalar("loss/train", trainer.latest_loss.item(), step)

        for index, optimizer in enumerate(trainer.optimizers):
            for group_index, group in enumerate(optimizer.raw_optimizer.param_groups):
                self.writer.add_scalar(
                    f"optimizer/{index}/lr/{group_index}",
                    group["lr"],
                    step,
                )

        if self.log_gradients:
            for name, parameter in trainer.named_parameters():
                if parameter.grad is not None:
                    self.writer.add_scalar(
                        f"gradients/{name}",
                        parameter.grad.detach().norm().item(),
                        step,
                    )

    def close(self):
        if self.writer is not None:
            self.writer.close()


# Kept as an alias for code written against the initial callback API. New code
# should use TensorBoardLogger; XAI is reserved for explainability callbacks.
XAI = TensorBoardLogger
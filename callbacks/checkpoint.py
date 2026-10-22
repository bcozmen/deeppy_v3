"""Periodic checkpoint callback."""

from .base import Callback


class CheckpointCallback(Callback):
    """Save a model checkpoint after every configured number of updates."""

    def __init__(self, every_n_steps=1000):
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be positive")
        self.every_n_steps = every_n_steps

    def on_optimizer_step_end(self, trainer, updated):
        if updated and trainer.checkpointer is not None:
            if trainer.global_step % self.every_n_steps == 0:
                trainer.checkpointer.save()

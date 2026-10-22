"""Abstract optimizer lifecycle used by the optimizer implementation.

This module defines the contract shared by the base PyTorch adapter and all
optimizer wrappers.
"""

from abc import ABC, abstractmethod
import torch


class BaseOptimizer(ABC):
    """Common interface for a trainable optimizer and its wrappers.

    The interface separates a training update into three phases:
    :meth:`prepare_step` prepares gradients, :meth:`update` attempts the
    parameter update, and :meth:`after_step` performs post-update work. The
    boolean returned by :meth:`update` allows wrappers such as AMP to report
    that an update was skipped because gradients were non-finite.
    """

    @property
    @abstractmethod
    def raw_optimizer(self):
        """Return the underlying ``torch.optim.Optimizer`` instance."""
        ...

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate ``loss`` using this optimizer's gradient policy."""
        ...

    def prepare_step(self) -> None:
        """Prepare gradients before the parameter update.

        Implementations may unscale gradients or clip them here.
        """
        pass

    @abstractmethod
    def update(self) -> bool:
        """Attempt a parameter update and return whether it succeeded."""
        ...

    def after_step(self, updated: bool) -> None:
        """Run post-update operations after an attempted update."""
        pass

    @abstractmethod
    def zero_grad(self) -> None:
        """Clear gradients accumulated on the model parameters."""
        ...

    def step(self) -> bool:
        """Run one complete prepare, update, and post-update cycle."""
        self.prepare_step()
        updated = self.update()
        self.after_step(updated)
        return updated

    def save_dict(self) -> dict:
        """Return this optimizer's complete state for checkpointing."""
        return self.state_dict()

    def load_dict(self, state: dict) -> None:
        """Restore this optimizer's complete state from ``state``."""
        self.load_state_dict(state)

    def state_dict(self) -> dict:
        """Return the underlying PyTorch optimizer state by default."""
        return self.raw_optimizer.state_dict()

    def load_state_dict(self, state: dict) -> None:
        """Load the underlying PyTorch optimizer state by default."""
        self.raw_optimizer.load_state_dict(state)

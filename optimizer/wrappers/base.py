"""Base delegation class for composable optimizer wrappers."""

from ..base import BaseOptimizer


class OptimizerWrapper(BaseOptimizer):
    """Delegate optimizer behavior while allowing wrappers to add a stage."""

    def __init__(self, inner: BaseOptimizer):
        """Wrap another optimizer-like object."""
        self.inner = inner

    @property
    def raw_optimizer(self):
        """Expose the original PyTorch optimizer through the wrapper chain."""
        return self.inner.raw_optimizer

    def backward(self, loss):
        """Delegate backward computation to the wrapped optimizer."""
        self.inner.backward(loss)

    def prepare_step(self):
        """Delegate gradient preparation to the wrapped optimizer."""
        self.inner.prepare_step()

    def update(self):
        """Delegate the parameter update and return its success status."""
        return self.inner.update()

    def after_step(self, updated):
        """Delegate post-update processing."""
        self.inner.after_step(updated)

    def zero_grad(self):
        """Delegate gradient clearing."""
        self.inner.zero_grad()

    def state_dict(self):
        """Return the wrapped optimizer's serializable state."""
        return self.inner.state_dict()

    def load_state_dict(self, state):
        """Restore the wrapped optimizer from a serialized state."""
        self.inner.load_state_dict(state)

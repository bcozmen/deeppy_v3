from .base import BaseOptimizer

class TorchOptimizer(BaseOptimizer):
    """Low-level adapter around one ``torch.optim.Optimizer`` instance."""

    def __init__(self, parameters, optimizer_cls, **kwargs):
        self._optimizer = optimizer_cls(parameters, **kwargs)

    @property
    def raw_optimizer(self):
        """Return the wrapped PyTorch optimizer."""
        return self._optimizer

    def backward(self, loss):
        """Compute ordinary unscaled gradients."""
        loss.backward()

    def update(self):
        """Apply one parameter update."""
        self._optimizer.step()
        return True

    def zero_grad(self):
        """Clear gradients on the optimized parameters."""
        self._optimizer.zero_grad(set_to_none=True)
"""Learning-rate scheduling wrapper for optimizer chains."""

from .base import OptimizerWrapper


class Scheduled(OptimizerWrapper):
    """Advance a PyTorch learning-rate scheduler after successful updates."""

    def __init__(self, inner, scheduler_cls, auto_step, **kwargs):
        """Create a scheduler for the raw PyTorch optimizer.

        Args:
            inner: Optimizer being wrapped.
            scheduler_cls: PyTorch scheduler class to instantiate.
            auto_step: If true, call the scheduler after each successful
                optimizer update.
            **kwargs: Keyword arguments forwarded to ``scheduler_cls``.
        """
        super().__init__(inner)

        self.auto_step = auto_step
        self.scheduler = scheduler_cls(self.raw_optimizer, **kwargs)

    def after_step(self, updated):
        """Run inner hooks and optionally advance the scheduler."""
        # Let inner wrappers finish their post-update processing.
        self.inner.after_step(updated)

        # Don't advance the scheduler if AMP skipped the update.
        if updated and self.auto_step:
            self.scheduler.step()

    def scheduler_step(self, metric=None):
        """Manually advance the scheduler, optionally using a metric."""
        if metric is None:
            self.scheduler.step()
        else:
            self.scheduler.step(metric)

    def state_dict(self):
        """Save wrapped optimizer and scheduler state."""
        return {
            "inner": self.inner.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state):
        """Restore wrapped optimizer and scheduler state."""
        self.inner.load_state_dict(state["inner"])
        self.scheduler.load_state_dict(state["scheduler"])

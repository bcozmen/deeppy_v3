"""High-level configuration facade for the optimizer stack."""

from contextlib import nullcontext

import torch
import torch.optim as optim

from .torch_optimizer import TorchOptimizer
from .gradient_accumulation import GradientAccumulation
from .wrappers.amp import AMP
from .wrappers.clipping import GradientClipping
from .wrappers.scheduler import Scheduled


def _feature_options(value, defaults, aliases=None):
    """Return normalized feature options, or ``None`` when disabled."""
    if value is None or value is False:
        return None

    options = {} if value is True else dict(value)
    for alias, name in (aliases or {}).items():
        if alias in options:
            if name in options:
                raise TypeError(f"use either {alias} or {name}, not both")
            options[name] = options.pop(alias)
    return {**defaults, **options}


class Optimizer:
    def __init__(self, parameters, optimizer_cls=optim.AdamW, *, runtime=None, clipping=None, scheduler=None, accumulation_steps=1, **optimizer_kwargs):
        """Build a selectively configured optimizer stack.

        Args:
            parameters: Iterable of model parameters. It is materialized once
                so the same parameters can be passed to the base optimizer
                and gradient clipping.
            optimizer_cls: PyTorch optimizer class.
            runtime: Shared :class:`Runtime` object. Its AMP configuration is
                used for scaling and autocasting.
            clipping: ``None`` or ``False`` disables clipping. A dictionary
                enables clipping and may contain ``max_value``, ``clip_type``,
                and ``norm_type``. ``max_norm`` is accepted as a friendly
                alias for ``max_value``.
            scheduler: ``None`` or ``False`` disables scheduling. A
                dictionary must contain ``scheduler_cls`` and may contain
                ``auto_step`` plus scheduler constructor arguments.
            accumulation_steps: Number of microsteps per update.
            **optimizer_kwargs: Arguments such as ``lr`` and ``weight_decay``
                forwarded to ``optimizer_cls``.
        """
        parameters = list(parameters)
        stack = TorchOptimizer(parameters, optimizer_cls, **optimizer_kwargs)

        amp_options = runtime.amp.options if runtime is not None else None
        self.amp = self._add_amp(stack, amp_options)
        if self.amp is not None:
            stack = self.amp

        self.clipping = self._add_clipping(stack, parameters, clipping)
        if self.clipping is not None:
            stack = self.clipping

        self.scheduler = self._add_scheduler(stack, scheduler)
        if self.scheduler is not None:
            stack = self.scheduler

        self.stack = stack
        self.accumulation = GradientAccumulation(stack, accumulation_steps)
        self.runtime = runtime

    @staticmethod
    def _add_amp(stack, config):
        """Add AMP when configured and return the resulting wrapper."""
        options = _feature_options(
            config,
            {"device_type": "cuda", "dtype": torch.float16, "enabled": True},
        )
        return AMP(stack, **options) if options is not None else None

    @staticmethod
    def _add_clipping(stack, parameters, config):
        """Add gradient clipping when configured and return its wrapper."""
        options = _feature_options(
            config,
            {"max_value": 1.0, "clip_type": "norm", "norm_type": 2.0},
            aliases={"max_norm": "max_value"},
        )
        return GradientClipping(stack, parameters, **options) if options is not None else None

    @staticmethod
    def _add_scheduler(stack, config):
        """Add a scheduler when configured and return its wrapper."""
        options = _feature_options(config, {"auto_step": True})
        if options is None:
            return None

        scheduler_cls = options.pop("scheduler_cls", None)
        if scheduler_cls is None:
            raise TypeError("scheduler requires scheduler_cls")
        return Scheduled(stack, scheduler_cls, **options)

    @property
    def raw_optimizer(self):
        """Return the underlying PyTorch optimizer."""
        return self.stack.raw_optimizer

    def autocast(self):
        """Return AMP's context, or a no-op context when AMP is disabled."""
        if self.runtime is not None:
            return self.runtime.autocast()
        if self.amp is None:
            return nullcontext()
        return self.amp.autocast()

    def backward(self, loss):
        """Accumulate a loss and update at the configured boundary."""
        return self.accumulation.backward(loss)

    def zero_grad(self):
        """Clear gradients and reset the accumulation counter."""
        self.stack.zero_grad()
        self.accumulation.microstep = 0

    def step(self):
        """Perform one immediate update without accumulation.

        Prefer :meth:`backward` for normal training when accumulation is
        enabled. This method is useful for explicit, already-backpropagated
        updates.
        """
        updated = self.stack.step()
        self.stack.zero_grad()
        self.accumulation.microstep = 0
        return updated

    def state_dict(self):
        """Save wrapper state and the current accumulation position."""
        return {
            "stack": self.stack.state_dict(),
            "microstep": self.accumulation.microstep,
        }

    def load_state_dict(self, state):
        """Restore wrapper state and the current accumulation position."""
        self.stack.load_state_dict(state["stack"])
        self.accumulation.microstep = state["microstep"]

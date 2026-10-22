"""Gradient clipping wrapper for optimizer chains."""

import torch

from .base import OptimizerWrapper


class GradientClipping(OptimizerWrapper):
    """Clip gradients after inner wrappers have prepared them."""

    def __init__(self, inner, parameters, max_value, clip_type, norm_type):
        """Configure norm- or value-based gradient clipping.

        Args:
            inner: Optimizer being wrapped.
            parameters: Model parameters whose gradients should be clipped.
            max_value: Maximum norm or absolute value, depending on
                ``clip_type``.
            clip_type: Either ``"norm"`` or ``"value"``.
            norm_type: Norm order used when ``clip_type`` is ``"norm"``.
        """
        super().__init__(inner)

        if clip_type not in {"norm", "value"}:
            raise ValueError("clip_type must be either 'norm' or 'value'")

        # Materialize generators so the parameters can be reused.
        self.parameters = list(parameters)
        self.max_value = max_value
        self.norm_type = norm_type
        self.clip_type = clip_type

    def prepare_step(self):
        """Prepare inner gradients, then clip the resulting gradients."""
        # First, allow AMP to unscale the gradients.
        self.inner.prepare_step()

        # Then clip the unscaled gradients.
        if self.clip_type == "norm":
            torch.nn.utils.clip_grad_norm_(self.parameters, max_norm=self.max_value, norm_type=self.norm_type)
        elif self.clip_type == "value":
            torch.nn.utils.clip_grad_value_(self.parameters, clip_value=self.max_value)
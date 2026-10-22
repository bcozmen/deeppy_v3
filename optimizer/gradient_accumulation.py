"""Controller for averaging gradients across multiple microsteps."""

class GradientAccumulation:
    """Accumulate gradients over several microsteps before updating.

    The loss is divided by ``accumulation_steps`` so the resulting gradient
    matches the average gradient of the complete accumulation group.
    """

    def __init__(self, optimizer, accumulation_steps=1):
        """Create a controller around an optimizer.

        Args:
            optimizer: Optimizer implementing ``backward``, ``step``, and
                ``zero_grad``.
            accumulation_steps: Number of microsteps per parameter update.
        """
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")

        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.microstep = 0
        self.loss_sum = None
        self.loss_count = 0
        self.last_loss = None

        self.optimizer.zero_grad()

    def backward(self, loss, metric_loss=None):
        """Accumulate one loss and update when the group is complete.

        ``metric_loss`` can be a detached, globally reduced loss used for
        reporting. The differentiable ``loss`` is still used for backward.

        Returns:
            ``False`` for an incomplete group, otherwise the optimizer's
            update status.
        """
        metric_loss = loss.detach() if metric_loss is None else metric_loss.detach()
        self.loss_sum = metric_loss if self.loss_sum is None else self.loss_sum + metric_loss
        self.loss_count += 1

        # Normalize the loss to obtain the average gradient
        # over the accumulation group.
        self.optimizer.backward(loss / self.accumulation_steps)
        self.microstep += 1

        if self.microstep < self.accumulation_steps:
            return False

        updated = self.optimizer.step()

        self.last_loss = self.loss_sum / self.loss_count

        self.optimizer.zero_grad()
        self.microstep = 0
        self.loss_sum = None
        self.loss_count = 0

        return updated

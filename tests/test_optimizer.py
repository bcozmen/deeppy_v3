"""Unit and integration tests for the high-level optimizer facade."""

import pytest
import torch

from optimizer.optimizer import Optimizer
from runtime import Runtime


def test_accumulation_matches_average_gradient():
    """Accumulation should match one update using the average loss gradient."""
    torch.manual_seed(11)
    accumulated_model = torch.nn.Linear(3, 2)
    reference_model = torch.nn.Linear(3, 2)
    reference_model.load_state_dict(accumulated_model.state_dict())

    inputs = [torch.randn(5, 3), torch.randn(5, 3)]
    targets = [torch.randn(5, 2), torch.randn(5, 2)]
    loss_fn = torch.nn.MSELoss()

    optimizer = Optimizer(
        accumulated_model.parameters(),
        optimizer_cls=torch.optim.SGD,
        lr=0.1,
        accumulation_steps=2,
    )
    for batch_inputs, batch_targets in zip(inputs, targets):
        loss = loss_fn(accumulated_model(batch_inputs), batch_targets)
        updated = optimizer.backward(loss)
    assert updated is True

    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1)
    reference_optimizer.zero_grad()
    for batch_inputs, batch_targets in zip(inputs, targets):
        loss = loss_fn(reference_model(batch_inputs), batch_targets) / 2
        loss.backward()
    reference_optimizer.step()

    for actual, expected in zip(accumulated_model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_training_reduces_loss_with_full_stack():
    """AMP, clipping, scheduling, and accumulation should train a CUDA model."""
    torch.manual_seed(23)
    device = torch.device("cuda")

    inputs = torch.randn(128, 4, device=device)
    target_weights = torch.tensor(
        [[1.5, -2.0, 0.5, 1.0], [-0.5, 1.0, 2.0, -1.5]],
        device=device,
    )
    targets = inputs @ target_weights.T

    model = torch.nn.Sequential(torch.nn.Linear(4, 12), torch.nn.ReLU(), torch.nn.Linear(12, 2)).to(device)
    loss_fn = torch.nn.MSELoss()
    runtime = Runtime(device, amp={"device_type": "cuda", "dtype": torch.float16})
    optimizer = Optimizer(
        model.parameters(),
        optimizer_cls=torch.optim.AdamW,
        lr=3e-3,
        runtime=runtime,
        clipping={"max_norm": 1.0},
        scheduler={
            "scheduler_cls": torch.optim.lr_scheduler.StepLR,
            "step_size": 20,
            "gamma": 0.5,
        },
        accumulation_steps=4,
    )

    first_loss = None
    last_loss = None
    for _ in range(8):
        order = torch.randperm(inputs.size(0), device=device)
        for start in range(0, inputs.size(0), 16):
            batch = order[start : start + 16]
            with optimizer.autocast():
                loss = loss_fn(model(inputs[batch]), targets[batch])
            if first_loss is None:
                first_loss = loss.detach().float().item()
            last_loss = loss.detach().float().item()
            optimizer.backward(loss)

    assert first_loss is not None and last_loss is not None
    assert torch.isfinite(torch.tensor(last_loss))
    assert last_loss < first_loss

    with torch.no_grad(), optimizer.autocast():
        final_loss = loss_fn(model(inputs), targets).float()
    assert torch.isfinite(final_loss)
    assert final_loss.item() < first_loss


def test_optional_features_can_be_disabled():
    """Each optional feature can be disabled without changing construction."""
    model = torch.nn.Linear(2, 1)
    optimizer = Optimizer(model.parameters(), lr=1e-2, clipping=None, scheduler=None)

    assert optimizer.amp is None
    assert optimizer.clipping is None
    assert optimizer.scheduler is None


def test_runtime_amp_can_be_explicitly_disabled():
    runtime = Runtime("cpu", amp={"enabled": False})

    assert runtime.amp.enabled is False
    assert runtime.amp.options["device_type"] == "cpu"


def test_invalid_clip_type_is_rejected():
    model = torch.nn.Linear(2, 1)

    with pytest.raises(ValueError, match="clip_type"):
        Optimizer(model.parameters(), clipping={"clip_type": "unknown"})

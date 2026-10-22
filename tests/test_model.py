"""Tests for runtime-aware model orchestration."""

import torch
import torch.nn as nn
import pytest

from callbacks import CheckpointCallback, TensorBoardLogger
from model import Model
from runtime import Runtime


class RecordingWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def close(self):
        pass


def test_model_builds_network_optimizer_and_checkpoint(tmp_path):
    runtime = Runtime("cpu", seed=7)
    logger = TensorBoardLogger(log_dir=str(tmp_path / "events"))
    model = Model(
        runtime,
        networks=[{"arch_params": {"layers": [3, 8, 1], "blocks": [nn.Linear, nn.ReLU]}}],
        optimizers=[{"lr": 0.05}],
        callbacks=[
            logger,
            CheckpointCallback(every_n_steps=1),
        ],
        checkpoint_dir=str(tmp_path / "checkpoint"),
    )

    batch = (torch.randn(8, 3), torch.randn(8, 1))
    updated, loss = model.optimize(batch)
    assert updated is True
    assert torch.isfinite(loss)

    model.checkpointer.save()
    assert (tmp_path / "checkpoint" / "checkpoint.pt").exists()
    assert (tmp_path / "checkpoint" / "rng_rank0.pt").exists()
    model.checkpointer.load()
    logger.close()
    assert list((tmp_path / "events").glob("events.out.tfevents.*"))


def test_tensorboard_logs_accumulated_mean_loss():
    writer = RecordingWriter()
    runtime = Runtime("cpu", seed=5)
    model = Model(
        runtime,
        networks=[{"arch_params": {"layers": [2, 1], "blocks": [nn.Linear]}}],
        optimizers=[{"lr": 0.1, "accumulation_steps": 2}],
        callbacks=[TensorBoardLogger(writer=writer)],
    )
    batch = (torch.ones(4, 2), torch.zeros(4, 1))

    _, first_loss = model.optimize(batch)
    _, second_loss = model.optimize(batch)

    logged_loss = [value for tag, value, _ in writer.scalars if tag == "loss/train"]
    assert len(logged_loss) == 1
    expected_loss = torch.stack([first_loss.detach(), second_loss.detach()]).mean().item()
    assert logged_loss[0] == pytest.approx(expected_loss)


def test_model_evaluate_does_not_update_parameters():
    runtime = Runtime("cpu")
    model = Model(
        runtime,
        networks=[{"arch_params": {"layers": [2, 1], "blocks": [nn.Linear]}}],
        optimizers=[{"lr": 0.1}],
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]
    model.evaluate((torch.randn(4, 2), torch.randn(4, 1)))
    after = list(model.parameters())

    for expected, actual in zip(before, after):
        torch.testing.assert_close(expected, actual)


def test_checkpoint_restores_model_optimizer_and_rng(tmp_path):
    runtime = Runtime("cpu", seed=17)
    model = Model(
        runtime,
        networks=[{"arch_params": {"layers": [2, 1], "blocks": [nn.Linear]}}],
        optimizers=[{"lr": 0.1}],
        checkpoint_dir=str(tmp_path / "checkpoint"),
    )
    batch = (torch.randn(4, 2), torch.randn(4, 1))
    model.optimize(batch)

    expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    expected_step = model.global_step
    model.checkpointer.save()
    expected_random_values = torch.rand(4)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    model.global_step = 99
    torch.rand(8)

    model.checkpointer.load()

    for expected, actual in zip(expected_parameters, model.parameters()):
        torch.testing.assert_close(expected, actual)
    assert model.global_step == expected_step
    torch.testing.assert_close(torch.rand(4), expected_random_values)


def test_checkpoint_allows_training_to_continue_identically(tmp_path):
    batch_one = (torch.tensor([[1.0, 2.0], [2.0, 1.0]]), torch.tensor([[1.0], [0.0]]))
    batch_two = (torch.tensor([[3.0, 1.0], [1.0, 3.0]]), torch.tensor([[0.0], [1.0]]))

    runtime = Runtime("cpu", seed=29)
    uninterrupted = Model(
        runtime,
        networks=[{
            "arch_params": {
                "layers": [2, 1],
                "blocks": [nn.Linear, nn.Dropout],
                "block_args": [{}, {"p": 0.5}],
            },
        }],
        optimizers=[{"lr": 0.05}],
        checkpoint_dir=str(tmp_path / "uninterrupted"),
    )
    uninterrupted.optimize(batch_one)
    uninterrupted.checkpointer.save()
    uninterrupted_loss = uninterrupted.optimize(batch_two)[1].detach().clone()
    uninterrupted_parameters = [parameter.detach().clone() for parameter in uninterrupted.parameters()]

    resumed_runtime = Runtime("cpu", seed=999)
    resumed = Model(
        resumed_runtime,
        networks=[{
            "arch_params": {
                "layers": [2, 1],
                "blocks": [nn.Linear, nn.Dropout],
                "block_args": [{}, {"p": 0.5}],
            },
        }],
        optimizers=[{"lr": 0.05}],
        checkpoint_dir=str(tmp_path / "uninterrupted"),
    )
    resumed.checkpointer.load()
    resumed_loss = resumed.optimize(batch_two)[1].detach().clone()

    torch.testing.assert_close(resumed_loss, uninterrupted_loss)
    for expected, actual in zip(uninterrupted_parameters, resumed.parameters()):
        torch.testing.assert_close(expected, actual)



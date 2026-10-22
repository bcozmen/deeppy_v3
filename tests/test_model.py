"""Tests for runtime-aware model orchestration."""

import torch
import torch.nn as nn
import pytest
from itertools import chain
from pathlib import Path

from callbacks import TensorBoardLogger
from model import BaseModel, Model
from network import Network
from optimizer import Optimizer
from runtime import Runtime


class RecordingWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def add_histogram(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def close(self):
        pass


class ThreeNetworkModel(BaseModel):
    """Build three branches from high-level dimensions and merge them."""

    def __init__(
        self,
        runtime,
        input_dim=4,
        hidden_dim=8,
        branch_1_dim=3,
        branch_2_dim=2,
        output_dim=1,
        lr=0.01,
    ):
        network_params = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "branch_1_dim": branch_1_dim,
            "branch_2_dim": branch_2_dim,
            "output_dim": output_dim,
        }
        super().__init__(
            runtime,
            network_params=network_params,
            optimizer_params={"lr": lr},
            model_params={**network_params, "lr": lr},
        )

    def build_networks(self, params):
        return [
            Network(
                {
                    "layers": [params["input_dim"], params["hidden_dim"], params["branch_1_dim"]],
                    "blocks": [nn.Linear, nn.ReLU],
                },
                runtime=self.runtime,
            ),
            Network(
                {
                    "layers": [params["input_dim"], params["hidden_dim"], params["branch_2_dim"]],
                    "blocks": [nn.Linear, nn.ReLU],
                },
                runtime=self.runtime,
            ),
            Network(
                {
                    "layers": [
                        params["branch_1_dim"] + params["branch_2_dim"],
                        params["hidden_dim"],
                        params["output_dim"],
                    ],
                    "blocks": [nn.Linear, nn.ReLU],
                },
                runtime=self.runtime,
            ),
        ]

    def build_optimizers(self, params):
        parameters = chain.from_iterable(network.parameters() for network in self.networks)
        return [Optimizer(parameters, runtime=self.runtime, **params)]

    def forward(self, inputs):
        branch_1 = self.networks[0](inputs)
        branch_2 = self.networks[1](inputs)
        if self.logger is not None:
            self.logger.log("network1/activation", branch_1)
        return self.networks[2](torch.cat([branch_1, branch_2], dim=-1))

    def training_step(self, batch):
        inputs, targets = batch
        return nn.functional.mse_loss(self(inputs), targets)


class TwoOptimizerModel(ThreeNetworkModel):
    def build_optimizers(self, params):
        return [
            Optimizer(self.networks[0].parameters(), runtime=self.runtime, **params),
            Optimizer(
                chain(self.networks[1].parameters(), self.networks[2].parameters()),
                runtime=self.runtime,
                **params,
            ),
        ]

    def optimize(self, batch):
        self.train()
        inputs, targets = self.runtime.to_device(batch)
        with self.runtime.autocast():
            branch_1 = self.networks[0](inputs)
            branch_2 = self.networks[1](inputs)
            loss_1 = branch_1.square().mean()
            prediction = self.networks[2](torch.cat([branch_1.detach(), branch_2], dim=-1))
            loss_2 = nn.functional.mse_loss(prediction, targets)
        updated_1 = self.optimizers[0].backward(loss_1, metric_loss=loss_1.detach())
        updated_2 = self.optimizers[1].backward(loss_2, metric_loss=loss_2.detach())
        loss = loss_1 + loss_2
        updated = updated_1 or updated_2
        if updated:
            self.global_step += 1
        return updated, loss


def test_model_builds_network_optimizer_and_checkpoint(tmp_path):
    runtime = Runtime("cpu", seed=7)
    logger = TensorBoardLogger(log_dir=str(tmp_path / "events"))
    model = Model(
        runtime,
        network_params=[{"arch_params": {"layers": [3, 8, 1], "blocks": [nn.Linear, nn.ReLU]}}],
        optimizer_params={"lr": 0.05},
    )
    checkpointer = runtime.create_checkpointer(str(tmp_path / "checkpoint"), model)

    batch = (torch.randn(8, 3), torch.randn(8, 1))
    updated, loss = model.optimize(batch)
    logger.on_optimizer_step_end(model, updated)
    assert updated is True
    assert torch.isfinite(loss)

    checkpointer.save()
    assert (tmp_path / "checkpoint" / "checkpoint.pt").exists()
    assert (tmp_path / "checkpoint" / "rng_rank0.pt").exists()
    checkpointer.load()
    logger.close()
    assert list((tmp_path / "events").glob("events.out.tfevents.*"))


def test_tensorboard_logs_accumulated_mean_loss():
    writer = RecordingWriter()
    runtime = Runtime("cpu", seed=5)
    model = Model(
        runtime,
        network_params=[{"arch_params": {"layers": [2, 1], "blocks": [nn.Linear]}}],
        optimizer_params={"lr": 0.1, "accumulation_steps": 2},
    )
    logger = TensorBoardLogger(writer=writer)
    batch = (torch.ones(4, 2), torch.zeros(4, 1))

    first_updated, first_loss = model.optimize(batch)
    logger.on_optimizer_step_end(model, first_updated)
    second_updated, second_loss = model.optimize(batch)
    logger.on_optimizer_step_end(model, second_updated)

    logged_loss = [value for tag, value, _ in writer.scalars if tag == "train/loss"]
    assert len(logged_loss) == 1
    expected_loss = torch.stack([first_loss.detach(), second_loss.detach()]).mean().item()
    assert logged_loss[0] == pytest.approx(expected_loss)


def test_model_can_log_intermediate_values_during_forward():
    writer = RecordingWriter()
    logger = TensorBoardLogger(writer=writer)
    model = ThreeNetworkModel(Runtime("cpu", seed=13)).attach_logger(logger)

    model.optimize((torch.randn(4, 4), torch.randn(4, 1)))

    assert any(tag == "train/network1/activation" for tag, _, _ in writer.scalars)


def test_tensorboard_output_is_stored_with_checkpoint():
    """Leave an inspectable TensorBoard run beside the checkpoint state."""
    run_path = Path(__file__).resolve().parents[1] / "test_outputs" / "model_run"
    checkpoint_path = run_path / "checkpoint"
    logs_path = checkpoint_path / "logs"

    runtime = Runtime("cpu", seed=19)
    model = Model(
        runtime,
        network_params=[
            {"arch_params": {"layers": [3, 8, 1], "blocks": [nn.Linear, nn.ReLU]}}
        ],
        optimizer_params={"lr": 0.05},
    )
    checkpointer = runtime.create_checkpointer(str(checkpoint_path), model)
    logger = TensorBoardLogger(log_dir=str(logs_path), log_gradients=True)

    updated, _ = model.optimize((torch.randn(8, 3), torch.randn(8, 1)))
    logger.on_before_optimizer_step(model)
    logger.on_optimizer_step_end(model, updated)
    checkpointer.save()
    logger.close()

    assert (checkpoint_path / "checkpoint.pt").exists()
    assert list(logs_path.glob("events.out.tfevents.*"))


def test_model_evaluate_does_not_update_parameters():
    runtime = Runtime("cpu")
    model = Model(
        runtime,
        network_params=[{"arch_params": {"layers": [2, 1], "blocks": [nn.Linear]}}],
        optimizer_params={"lr": 0.1},
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
        network_params=[{"arch_params": {"layers": [2, 1], "blocks": [nn.Linear]}}],
        optimizer_params={"lr": 0.1},
    )
    checkpointer = runtime.create_checkpointer(str(tmp_path / "checkpoint"), model)
    batch = (torch.randn(4, 2), torch.randn(4, 1))
    model.optimize(batch)

    expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    expected_step = model.global_step
    checkpointer.save()
    expected_random_values = torch.rand(4)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    model.global_step = 99
    torch.rand(8)

    checkpointer.load()

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
        network_params=[{
            "arch_params": {
                "layers": [2, 1],
                "blocks": [nn.Linear, nn.Dropout],
                "block_args": [{}, {"p": 0.5}],
            },
        }],
        optimizer_params={"lr": 0.05},
    )
    uninterrupted_checkpointer = runtime.create_checkpointer(
        str(tmp_path / "uninterrupted"), uninterrupted
    )
    uninterrupted.optimize(batch_one)
    uninterrupted_checkpointer.save()
    uninterrupted_loss = uninterrupted.optimize(batch_two)[1].detach().clone()
    uninterrupted_parameters = [parameter.detach().clone() for parameter in uninterrupted.parameters()]

    resumed_runtime = Runtime("cpu", seed=999)
    resumed = Model(
        resumed_runtime,
        network_params=[{
            "arch_params": {
                "layers": [2, 1],
                "blocks": [nn.Linear, nn.Dropout],
                "block_args": [{}, {"p": 0.5}],
            },
        }],
        optimizer_params={"lr": 0.05},
    )
    resumed_checkpointer = resumed_runtime.create_checkpointer(
        str(tmp_path / "uninterrupted"), resumed
    )
    resumed_checkpointer.load()
    resumed_loss = resumed.optimize(batch_two)[1].detach().clone()

    torch.testing.assert_close(resumed_loss, uninterrupted_loss)
    for expected, actual in zip(uninterrupted_parameters, resumed.parameters()):
        torch.testing.assert_close(expected, actual)


def test_model_can_be_initialized_directly_from_checkpoint(tmp_path):
    runtime = Runtime("cpu", seed=41)
    model = Model(
        runtime,
        network_params=[{"arch_params": {"layers": [2, 1], "blocks": [nn.Linear]}}],
        optimizer_params={"lr": 0.1},
    )
    model.optimize((torch.randn(3, 2), torch.randn(3, 1)))
    state = model.save_states()
    checkpoint_path = tmp_path / "model.pt"
    torch.save(state, checkpoint_path)

    restored = Model.from_checkpoint(Runtime("cpu", seed=99), checkpoint_path)

    assert restored.network_params == model.network_params
    assert restored.optimizer_params == model.optimizer_params
    assert restored.global_step == model.global_step
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)


def test_custom_model_builds_three_networks_with_one_optimizer():
    runtime = Runtime("cpu", seed=43)
    model = ThreeNetworkModel(
        runtime,
    )

    updated, loss = model.optimize((torch.randn(6, 4), torch.randn(6, 1)))

    assert len(model.networks) == 3
    assert len(model.optimizers) == 1
    assert updated is True
    assert torch.isfinite(loss)

    restored = ThreeNetworkModel.from_checkpoint(
        Runtime("cpu", seed=99),
        model.save_states(),
    )
    assert restored.model_params == model.model_params
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)


def test_custom_model_can_use_two_optimizers():
    model = TwoOptimizerModel(
        Runtime("cpu", seed=47),
    )

    updated, loss = model.optimize((torch.randn(6, 4), torch.randn(6, 1)))

    assert len(model.optimizers) == 2
    assert updated is True
    assert torch.isfinite(loss)


def test_base_model_is_abstract():
    with pytest.raises(TypeError):
        BaseModel(Runtime("cpu"))



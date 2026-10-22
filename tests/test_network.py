"""Unit tests for the modular network facade."""

import torch
import torch.nn as nn

from network import Network


def architecture():
    return {"layers": [4, 8, 2], "blocks": [nn.Linear, nn.ReLU]}


def test_regression_forward_shape():
    model = Network(architecture(), task="reg")
    outputs = model(torch.randn(6, 4))

    assert outputs.shape == (6, 2)
    assert torch.isfinite(outputs).all()


def test_classification_threshold_only_in_evaluation():
    model = Network(architecture(), task="classify", classify_threshold=0.25)
    inputs = torch.randn(6, 4)

    model.train()
    training_outputs = model(inputs)
    torch.testing.assert_close(training_outputs, model.model.model(inputs))

    model.eval()
    evaluation_outputs = model(inputs)
    assert set(torch.unique(evaluation_outputs).tolist()).issubset({0.0, 1.0})


def test_autoencoder_forward_encode_and_decode():
    model = Network(
        {"layers": [4, 2], "blocks": [nn.Linear]},
        decoder_params={"layers": [2, 4], "blocks": [nn.Linear]},
        task="autoencoder",
    )
    inputs = torch.randn(5, 4)
    encoded = model.encode(inputs)

    assert encoded.shape == (5, 2)
    assert model.decode(encoded).shape == inputs.shape
    assert model(inputs).shape == inputs.shape


def test_autoencoder_derives_symmetric_decoder():
    model = Network(
        {"layers": [4, 8, 2], "blocks": [nn.Linear, nn.ReLU]},
        task="autoencoder",
    )

    assert model.decoder_params[0]["layers"] == [2, 8, 4]
    assert model(torch.randn(5, 4)).shape == (5, 4)


def test_state_round_trip():
    torch.manual_seed(31)
    original = Network(architecture(), task="reg")
    state = original.save_states()
    restored = Network(architecture(), task="reg")
    restored.load_states(state)

    inputs = torch.randn(3, 4)
    torch.testing.assert_close(original(inputs), restored(inputs))


def test_compile_and_ddp_are_disabled_by_default():
    model = Network(architecture())

    assert model.model.__class__.__name__ == "TaskModel"
    assert not hasattr(model.model, "module")

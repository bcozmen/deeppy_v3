"""Declarative layer and residual-block generation."""

import torch.nn as nn


class ResidualBlock(nn.Module):
    """Add a block's output to its input."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, inputs):
        return self.layer(inputs) + inputs


class LayerGenerator:
    """Build sequential layers from architecture dictionaries.

    A specification has the same convenient shape as the legacy network API::

        {
            "layers": [32, 64, 10],
            "blocks": [nn.Linear, nn.ReLU],
            "block_args": [{}, {}],
            "out_act": nn.Identity,
        }
    """

    DIMENSIONAL_LAYERS = {
        "Bilinear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d",
        "ConvTranspose2d", "ConvTranspose3d", "Linear",
    }

    def generate(self, specification):
        layers = specification.get("layers", [None, None])
        blocks = specification.get("blocks", [])
        block_args = specification.get("block_args", [])
        out_act = specification.get("out_act", nn.Identity)
        out_params = specification.get("out_params", {})
        residual = specification.get("residual", False)

        if out_act is None and layers:
            raise ValueError("out_act cannot be None; use nn.Identity instead")

        block_args = list(block_args)
        block_args.extend({} for _ in range(len(blocks) - len(block_args)))
        generated = []
        for index, (inputs, outputs) in enumerate(zip(layers[:-1], layers[1:])):
            final = index == len(layers) - 2
            block = self._generate_block(
                blocks,
                block_args,
                inputs,
                outputs,
                out_act if final else None,
                out_params if final else {},
            )
            if residual:
                block = [ResidualBlock(nn.Sequential(*block))]
            generated.extend(block)
        return generated

    def _generate_block(self, blocks, block_args, inputs, outputs, out_act, out_params):
        generated = []
        dimensional_layer_created = False

        for block, arguments in zip(blocks, block_args):
            arguments = dict(arguments)
            name = block.__name__

            if inputs is not None and outputs is not None:
                if name in self.DIMENSIONAL_LAYERS:
                    generated.append(block(inputs, outputs, **arguments))
                    dimensional_layer_created = True
                    continue

                if name in self._activation_names():
                    if out_act is not None:
                        block, arguments = out_act, dict(out_params)
                    generated.append(block(**arguments))
                    continue

                if self._is_normalization(block):
                    feature_size = outputs if dimensional_layer_created else inputs
                    if name == "LayerNorm":
                        arguments.setdefault("normalized_shape", feature_size)
                    else:
                        arguments.setdefault("num_features", feature_size)

            generated.append(block(**arguments))
        return generated

    @staticmethod
    def _activation_names():
        return set(nn.modules.activation.__all__) | {"Identity"}

    @staticmethod
    def _is_normalization(block):
        return block.__name__ in nn.modules.batchnorm.__all__ or "InstanceNorm" in block.__name__ or block.__name__ == "LayerNorm"

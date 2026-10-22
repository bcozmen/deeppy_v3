# Modular network builder

The `network` package is a new implementation and does not depend on `v2` or
`env_config`.

`Network` is the high-level controller. It builds a model from layer
specifications and optionally applies weight initialization, `torch.compile`,
and `DistributedDataParallel` wrapping.

```python
import torch
import torch.nn as nn

from network import Network
from runtime import Runtime

runtime = Runtime("cuda", compile=False)

network = Network(
    {
        "layers": [32, 64, 10],
        "blocks": [nn.Linear, nn.ReLU],
    },
    task="reg",
    weight_init="uniform",
    runtime=runtime,
)

outputs = network(torch.randn(8, 32))
```

Available task implementations are regression, classification, and
autoencoder. Classification returns thresholded values only in evaluation
mode. Autoencoders expose `encode()` and `decode()`. If `decoder_params` is
omitted for an autoencoder, the controller creates it by reversing the layer
dimensions and architecture-specification order:

```python
network = Network(
    {"layers": [32, 64, 10], "blocks": [nn.Linear, nn.ReLU]},
    task="autoencoder",
)
# Decoder layers become [10, 64, 32].
```

Initialize `Runtime` once and pass it to the network. It owns device
placement, process-group setup, compilation, and DDP wrapping:

```python
from runtime import Runtime

runtime = Runtime(torch.device("cuda"))
network = Network(
    {"layers": [32, 64, 10], "blocks": [nn.Linear, nn.ReLU]},
    runtime=runtime,
)
```

The controller is split into focused components:

- `LayerGenerator` builds sequential layers.
- `WeightInitializer` initializes supported learnable layers.
- Runtime compilation and DDP are centralized in `runtime.Compiler` and
    `runtime.Distributed`.
- Task classes provide task-specific forward behavior.

## Tests

Run the complete optimizer and network suite from the repository root:

```text
python test.py
```

This includes unit tests, the CUDA integration test when CUDA is available,
and two-process CPU DDP tests for both the optimizer and network controller.

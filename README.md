# DeepPy v3

DeepPy v3 is a PyTorch training toolkit organized around two new composable
APIs:

- `runtime` — shared device, DDP, AMP, compilation, RNG, and checkpoint setup.
- `model` — runtime-aware model orchestration with networks, optimizers,
    checkpointing, and callbacks.
- `network` — declarative model construction for regression, classification,
  and autoencoder tasks.
- `optimizer` — composable optimizer, AMP scaling, clipping, scheduling, and
  gradient accumulation.
- `callbacks` — training lifecycle callbacks, TensorBoard logging, and
    periodic checkpointing.

The `v2` directory is the legacy implementation and is not used by the new
APIs.

## Quick start

```python
import torch
from network import Network
from optimizer import Optimizer
from runtime import Runtime

runtime = Runtime(
    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    amp={"device_type": "cuda", "dtype": torch.float16}
    if torch.cuda.is_available()
    else None,
    seed=42,
)

model = Network(
    {"layers": [4, 32, 2], "blocks": [torch.nn.Linear, torch.nn.ReLU]},
    task="reg",
    runtime=runtime,
)
optimizer = Optimizer(model.parameters(), lr=1e-3, runtime=runtime)

for inputs, targets in loader:
    with runtime.autocast():
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
    if optimizer.backward(loss):
        print("parameters updated")
```

## Model orchestration

For a standard supervised model, `Model` creates the configured networks and
optimizers and exposes one training operation:

```python
from callbacks import TensorBoardLogger
from model import ExampleModel

model = ExampleModel(
    runtime,
    network_params=[{
        "arch_params": {
            "layers": [4, 32, 2],
            "blocks": [torch.nn.Linear, torch.nn.ReLU],
        },
    }],
    optimizer_params={"lr": 1e-3},
)

updated, loss = model.optimize((inputs, targets))
logger = TensorBoardLogger(log_dir="runs/example")
logger.on_optimizer_step_end(model, updated)
checkpointer = runtime.create_checkpointer("checkpoints/example", model)
if updated:
    checkpointer.save()
```

`TensorBoardLogger` and the checkpointer are run-level services. A trainer or
`LearnFrame` should invoke them after `model.optimize()`. The checkpoint
service stores network, optimizer, dataset, and per-rank RNG state by
delegating RNG persistence to `Runtime`.
Use a separate XAI callback for actual explainability visualizations.

## Distributed training

Initialize `Runtime` once. It configures the local device and wraps the model
with DDP when launched with multiple processes:

```text
torchrun --standalone --nproc_per_node=2 train.py
```

No separate device or DDP arguments are passed to `Network`, and no AMP
arguments are passed to `Optimizer`. Both use the same runtime object.

## Network construction

Architecture specifications contain layer dimensions and block classes:

```python
architecture = {
    "layers": [16, 64, 8],
    "blocks": [torch.nn.Linear, torch.nn.ReLU],
}
```

Autoencoders can omit `decoder_params`; the network creates a symmetric
architecture automatically by reversing the dimensions and specification
order.

Additional network features include:

- Separate task classes for regression, classification, and autoencoding.
- Weight initialization strategies.
- Optional `torch.compile` through `Runtime`.
- Optional DDP through `Runtime`.

See [network/README.md](network/README.md) for details.

## Optimizer construction

```python
optimizer = Optimizer(
    model.parameters(),
    optimizer_cls=torch.optim.AdamW,
    lr=1e-4,
    runtime=runtime,
    clipping={"max_norm": 1.0},
    scheduler={
        "scheduler_cls": torch.optim.lr_scheduler.StepLR,
        "step_size": 10,
        "gamma": 0.1,
    },
    accumulation_steps=4,
)
```

See [optimizer/README.md](optimizer/README.md) for lifecycle and checkpoint
details.

## Testing

Run the complete test suite from the repository root:

```text
python test.py
```

This performs compilation checks, unit tests, CUDA integration tests when
CUDA is available, and two-process CPU DDP tests for both the network and
optimizer APIs.

The CUDA test suite includes a single-GPU runtime smoke test. The multi-GPU
DDP test is present but disabled until multi-GPU hardware is available; enable
`ENABLE_MULTI_GPU_DDP` in [tests/test_gpu_ddp.py](tests/test_gpu_ddp.py) and
launch it with `torchrun` when needed.

## Project layout

- [runtime](runtime) — shared execution environment.
- [model](model) — model and training-step orchestration.
- [network](network) — new modular network builder.
- [optimizer](optimizer) — new optimizer stack.
- [callbacks](callbacks) — lifecycle and TensorBoard callbacks.
- [tests](tests) — unit and DDP integration tests.
- [v2](v2) — legacy code kept for reference.

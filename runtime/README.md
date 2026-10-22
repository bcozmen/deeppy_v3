# Runtime

`Runtime` is the shared execution configuration for the new model and
optimizer APIs. Initialize it once at the beginning of a training program:

```python
import torch
from runtime import Runtime

runtime = Runtime(
    torch.device("cuda"),
    amp={"device_type": "cuda", "dtype": torch.float16},
    compile=False,
    seed=42,
)
```

Pass the same object to the network and optimizer:

```python
from network import Network
from optimizer import Optimizer

model = Network(architecture, runtime=runtime)
optimizer = Optimizer(model.parameters(), lr=1e-4, runtime=runtime)

with runtime.autocast():
    loss = loss_fn(model(inputs), targets)
updated = optimizer.backward(loss)
```

The runtime owns:

- Device selection and local-rank handling.
- Process-group initialization and DDP wrapping.
- AMP autocasting configuration.
- Optional `torch.compile` configuration.
- Process-wide Python, NumPy, and PyTorch RNG state.

The RNG is initialized from `seed` before models and optimizers are
constructed, so parameter initialization is reproducible. Checkpointing
remains a separate service, but uses `runtime.rng` by default.

The runtime also owns rank-specific RNG persistence:

```python
runtime.save_rng_state("checkpoints/example")
runtime.load_rng_state("checkpoints/example")
```

Normally `Checkpointer` calls these methods automatically. It does not accept
or manage a separate RNG object.

## Distributed training

Launch with `torchrun`; the runtime reads `LOCAL_RANK`, `RANK`, and
`WORLD_SIZE` automatically:

```text
torchrun --standalone --nproc_per_node=2 train.py
```

The same code works for one process or multiple processes. Call
`runtime.cleanup()` at the end of a distributed program.

## Components

- `Runtime` coordinates the execution setup.
- `Distributed` handles device placement and DDP.
- `AMPConfig` provides the autocast context.
- `Compiler` applies `torch.compile` when enabled.
- `RandomSeed` manages reproducible RNG state and is available as
    `runtime.rng`.
- `Checkpointer` provides checkpoint persistence.

`Runtime` intentionally does not own log directories or model checkpoints:
those belong to an individual training run rather than to the process
execution environment.

For metrics, use `runtime.reduce_mean(value)` when only a scalar aggregate is
needed. Use `runtime.gather(values)` when per-sample values must be preserved:

```python
per_sample_loss = torch.nn.functional.mse_loss(
    predictions,
    targets,
    reduction="none",
).flatten(1).mean(1)
all_losses = runtime.gather(per_sample_loss)
```

With four ranks and a local batch size of 32, `all_losses` contains up to 128
values. `gather()` also trims uneven local batches correctly, making it useful
for entropy, percentile, histogram, and other custom TensorBoard statistics.

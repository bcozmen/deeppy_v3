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

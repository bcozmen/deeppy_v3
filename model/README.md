# Model orchestration

`Model` is the high-level training facade for the new APIs. It receives one
shared `Runtime`, builds configured networks and optimizers, and coordinates
training callbacks and checkpointing.

```python
import torch
from callbacks import CheckpointCallback, TensorBoardLogger
from model import Model
from runtime import Runtime

runtime = Runtime(
    "cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
)

model = Model(
    runtime,
    networks=[{
        "arch_params": {
            "layers": [4, 32, 2],
            "blocks": [torch.nn.Linear, torch.nn.ReLU],
        },
    }],
    optimizers=[{"lr": 1e-3}],
    callbacks=[
        TensorBoardLogger("runs/example"),
        CheckpointCallback(every_n_steps=1000),
    ],
    checkpoint_dir="checkpoints/example",
)

updated, loss = model.optimize((inputs, targets))
```

`Runtime` owns execution concerns such as device placement, DDP, AMP,
compilation, and RNG. Checkpointing and logging are run-level services that
are attached to the model through callbacks or `checkpoint_dir`.

`TensorBoardLogger` is intentionally separate from XAI. It records training
metrics; an actual XAI callback can later record attributions, embeddings, or
other model explanations.

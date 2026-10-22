# Model orchestration

`BaseModel` is an abstract training facade. Subclasses define
`build_networks()`, `build_optimizers()`, `forward()`, and `training_step()`.
The base class owns the stable training lifecycle: device transfer,
autocasting, backward/update handling, evaluation, and state serialization.
`ExampleModel` demonstrates a simple implementation.

A subclass can expose high-level arguments instead of requiring callers to
provide architecture dictionaries directly:

```python
class CustomModel(BaseModel):
    def __init__(self, runtime, latent_dim, hidden_dim, num_layers, lr):
        network_params = {
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }
        super().__init__(
            runtime,
            network_params=network_params,
            optimizer_params={"lr": lr},
            model_params={**network_params, "lr": lr},
        )

    def build_networks(self, params):
        # Construct the actual Network configurations here.
        ...
```

`model_params` is stored in the checkpoint, allowing
`CustomModel.from_checkpoint(runtime, checkpoint)` to call the custom
constructor with the original high-level arguments.

```python
import torch
from callbacks import TensorBoardLogger
from model import ExampleModel
from runtime import Runtime

runtime = Runtime(
    "cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
)

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
logger = TensorBoardLogger("runs/example")
logger.on_optimizer_step_end(model, updated)
checkpointer = runtime.create_checkpointer("checkpoints/example", model)
if updated:
    checkpointer.save()
```

`Runtime` owns execution concerns such as device placement, DDP, AMP,
compilation, RNG, and recursive batch device transfer. Checkpointing and
logging are run-level services owned by the trainer or `LearnFrame`.

`TensorBoardLogger` is intentionally separate from XAI. It records training
metrics; an actual XAI callback can later record attributions, embeddings, or
other model explanations.

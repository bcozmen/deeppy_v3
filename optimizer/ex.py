import torch
import torch.nn as nn

from .optimizer import Optimizer
from runtime import Runtime


model = MyModel()  # Replace with the model being trained.
loss_fn = nn.CrossEntropyLoss()
runtime = Runtime("cuda", amp={"device_type": "cuda", "dtype": torch.float16})
model = runtime.distributed.wrap_model(model)

optimizer = Optimizer(
    model.parameters(),
    optimizer_cls=torch.optim.AdamW,
    lr=1e-4,
    runtime=runtime,
    clipping={
        "max_norm": 1.0,
    },
    scheduler={
        "scheduler_cls": torch.optim.lr_scheduler.StepLR,
        "step_size": 10,
        "gamma": 0.1,
    },
    accumulation_steps=4,
)

for inputs, targets in loader:  # Replace with the project's dataloader.
    with runtime.autocast():
        logits = model(inputs)
        loss = loss_fn(logits, targets)

    # False means the accumulation group is not complete, or AMP skipped the
    # update. True means parameters were updated successfully.
    updated = optimizer.backward(loss)

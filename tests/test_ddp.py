"""Two-process CPU DDP integration test.

Run with:
    torchrun --standalone --nproc_per_node=2 tests/test_ddp.py
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from runtime import Distributed
from optimizer.optimizer import Optimizer


def main():
    runtime = Distributed(torch.device("cpu"))
    if runtime.world_size < 2:
        raise RuntimeError("run this test with torchrun --nproc_per_node=2")

    try:
        torch.manual_seed(101)
        model = runtime.wrap_model(nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2)))
        optimizer = Optimizer(
            model.parameters(),
            optimizer_cls=torch.optim.SGD,
            lr=0.1,
            clipping={"max_norm": 1.0},
            scheduler={
                "scheduler_cls": torch.optim.lr_scheduler.StepLR,
                "step_size": 1,
                "gamma": 0.9,
            },
            accumulation_steps=2,
        )
        loss_fn = nn.MSELoss()
        updates = 0

        for step in range(4):
            generator = torch.Generator().manual_seed(1000 + runtime.rank * 10 + step)
            inputs = torch.randn(8, 4, generator=generator)
            targets = torch.randn(8, 2, generator=generator)
            loss = loss_fn(model(inputs), targets)
            if optimizer.backward(loss):
                updates += 1

        assert updates == 2
        assert abs(optimizer.raw_optimizer.param_groups[0]["lr"] - 0.081) < 1e-7

        parameters = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
        minimum = parameters.clone()
        maximum = parameters.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        torch.testing.assert_close(minimum, maximum, rtol=0.0, atol=1e-6)

        runtime.barrier()
        if runtime.is_main_process:
            print("torchrun CPU DDP optimizer test passed")
    finally:
        runtime.cleanup()


if __name__ == "__main__":
    main()

"""Two-process CPU DDP test for the network facade.

Run with:
    torchrun --standalone --nproc_per_node=2 tests/test_network_ddp.py
"""

import torch
import torch.nn as nn

from network import Network
from runtime import Runtime


def main():
    runtime = Runtime(torch.device("cpu"))
    try:
        model = Network(
            {"layers": [4, 8, 2], "blocks": [nn.Linear, nn.ReLU]},
            runtime=runtime,
        )
        outputs = model(torch.randn(8, 4))
        outputs.square().mean().backward()

        parameters = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
        minimum = parameters.clone()
        maximum = parameters.clone()
        torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
        torch.testing.assert_close(minimum, maximum, rtol=0.0, atol=1e-6)

        runtime.barrier()
        if runtime.is_main_process:
            print("torchrun CPU network DDP test passed")
    finally:
        runtime.cleanup()


if __name__ == "__main__":
    main()

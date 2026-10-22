"""Optional CUDA runtime and DDP smoke tests.

The multi-GPU case is deliberately disabled until it can be run on hardware
with more than one visible GPU. Enable it and launch this module with
``torchrun --nproc_per_node=<count>`` when that hardware is available.
"""

import os

import pytest
import torch
import torch.nn as nn

from network import Network
from runtime import Runtime


ENABLE_MULTI_GPU_DDP = False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_single_gpu_runtime_path():
    runtime = Runtime("cuda")
    network = Network(
        {"layers": [3, 4, 1], "blocks": [nn.Linear, nn.ReLU]},
        runtime=runtime,
    )

    output = network(torch.randn(2, 3, device=runtime.device))

    assert runtime.world_size == 1
    assert output.device == runtime.device


@pytest.mark.skipif(
    not ENABLE_MULTI_GPU_DDP,
    reason="multi-GPU DDP test is disabled until multi-GPU hardware is available",
)
def test_multi_gpu_ddp_path():
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("launch this test with torchrun and at least two processes")

    runtime = Runtime("cuda")
    network = Network(
        {"layers": [3, 4, 1], "blocks": [nn.Linear, nn.ReLU]},
        runtime=runtime,
    )
    output = network(torch.randn(2, 3, device=runtime.device))

    assert runtime.world_size >= 2
    assert output.device == runtime.device


if __name__ == "__main__":
    test_single_gpu_runtime_path()
    print("single-GPU CUDA runtime test passed")

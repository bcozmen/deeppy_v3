import os
import random
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.data.distributed import DistributedSampler


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEVICE = "cuda"

SEED = 42

EPOCHS = 10
LOCAL_BATCH_SIZE = 32
ACCUMULATION_STEPS = 4

LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 1.0

NUM_WORKERS = 2

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "latest.pt"

USE_AMP = True
USE_CUDA_PREFETCH = True

# Choose "auto", "bf16", or "fp16".
AMP_PRECISION = "auto"

# Enable only if strict deterministic execution is required.
STRICT_DETERMINISM = False


if DEVICE == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank >= torch.cuda.device_count():
        raise RuntimeError("LOCAL_RANK exceeds visible GPU count.")

    torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)
    backend = "nccl"

elif DEVICE == "cpu":
    local_rank = 0

    device = torch.device("cpu")
    backend = "gloo"

else:
    raise ValueError("DEVICE must be 'cuda' or 'cpu'.")


dist.init_process_group(backend=backend)

rank = dist.get_rank()
world_size = dist.get_world_size()

is_main_process = rank == 0

if is_main_process:
    print(f"World size: {world_size}")
    print(f"Device: {device}")



# ============================================================
# 4. DATASET
# ============================================================

# DEMONSTRATION ONLY:
# Replace this section with your real dataset.
#
# For a large real dataset, don't broadcast the entire dataset.
# Load it using your actual dataset implementation instead.

seed_everything(SEED)

if rank == 0:
    X = torch.randn(10000, 10)
    Y = torch.randn(10000, 5)
else:
    X = torch.empty(10000, 10)
    Y = torch.empty(10000, 5)

dist.broadcast(X, src=0)
dist.broadcast(Y, src=0)

dataset = TensorDataset(X, Y)


# ============================================================
# 5. CUSTOM SAMPLE SELECTION + DISTRIBUTED SAMPLING
# ============================================================

def select_training_indices(dataset):
    """
    Replace this function with your custom sample-selection logic.

    Examples:
      - Filter by class.
      - Select samples from certain groups.
      - Exclude invalid samples.
      - Return indices selected by an external algorithm.

    This function should produce the same eligible indices on
    every rank for this particular implementation.
    """

    return list(range(len(dataset)))


selected_indices = select_training_indices(dataset)

# First choose which samples are eligible.
selected_dataset = Subset(dataset, selected_indices)

# Then distribute those samples across ranks.
sampler = DistributedSampler(
    selected_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=SEED,
    drop_last=True,
)

loader = DataLoader(
    selected_dataset,
    batch_size=LOCAL_BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
    drop_last=True,
    persistent_workers=(NUM_WORKERS > 0),
)


# ============================================================
# 6. THREE INDEPENDENT NETWORKS
# ============================================================

# Same seed on every rank before model initialization.
seed_everything(SEED)


net1 = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 5),
).to(device)


net2 = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 5),
).to(device)


net3 = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
).to(device)


# Each network has its own DDP wrapper.

if DEVICE == "cuda":
    net1 = DDP(net1, device_ids=[local_rank])
    net2 = DDP(net2, device_ids=[local_rank])
    net3 = DDP(net3, device_ids=[local_rank])
else:
    net1 = DDP(net1)
    net2 = DDP(net2)
    net3 = DDP(net3)


models = [net1, net2, net3]

parameters = [
    parameter
    for model in models
    for parameter in model.parameters()
]


# Use one optimizer for all three networks.

optimizer = torch.optim.Adam(
    parameters,
    lr=LEARNING_RATE,
)


# ============================================================
# 7. AMP CONFIGURATION
# ============================================================

if DEVICE == "cuda" and USE_AMP:

    if AMP_PRECISION == "auto":
        amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

    elif AMP_PRECISION == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not supported by this GPU.")

        amp_dtype = torch.bfloat16

    elif AMP_PRECISION == "fp16":
        amp_dtype = torch.float16

    else:
        raise ValueError("Invalid AMP_PRECISION.")

    amp_enabled = True

else:
    amp_dtype = torch.float32
    amp_enabled = False


# FP16 generally needs gradient scaling; BF16 usually doesn't.

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(
        amp_enabled
        and amp_dtype == torch.float16
    ),
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)


# ============================================================
# 8. CUDA INPUT PREFETCHER
# ============================================================

class CUDAPrefetcher:
    """
    Prefetch the next batch on a dedicated CUDA stream.

    The DataLoader should use pin_memory=True.

    This overlaps host-to-device transfers with computation
    when the hardware and workload permit it.
    """

    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device

        self.stream = torch.cuda.Stream(device=device)

        self.next_batch = None

        self._preload()

    def _preload(self):
        try:
            x, target = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            x = x.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            self.next_batch = (x, target)

    def __iter__(self):
        return self

    def __next__(self):

        if self.next_batch is None:
            raise StopIteration

        current_stream = torch.cuda.current_stream(self.device)

        # Wait until the prefetched batch is ready.
        current_stream.wait_stream(self.stream)

        batch = self.next_batch

        # Ensure tensors are kept alive until computation
        # on the current stream has finished using them.
        for tensor in batch:
            tensor.record_stream(current_stream)

        # Begin prefetching the following batch.
        self._preload()

        return batch


def make_batch_iterator(loader):
    if DEVICE == "cuda" and USE_CUDA_PREFETCH:
        return CUDAPrefetcher(loader, device)

    return iter(loader)


# ============================================================
# 9. CHECKPOINTING
# ============================================================

def save_checkpoint(
    epoch,
    global_step,
    rank_rng_states,
):
    """
    All ranks must call this function because gathering the RNG
    states is a collective operation.

    Only rank 0 writes the checkpoint.
    """

    if is_main_process:

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,

            "net1": net1.module.state_dict(),
            "net2": net2.module.state_dict(),
            "net3": net3.module.state_dict(),

            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),

            "rng_states": rank_rng_states,

            "seed": SEED,
            "world_size": world_size,
        }

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        temporary_path = CHECKPOINT_DIR / "latest.tmp"

        torch.save(checkpoint, temporary_path)

        # Avoid leaving a partially written latest.pt.
        os.replace(temporary_path, CHECKPOINT_PATH)

        print(f"Saved checkpoint: {CHECKPOINT_PATH}")


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return 0, 0

    # This is a trusted checkpoint created by this script.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    net1.module.load_state_dict(checkpoint["net1"])
    net2.module.load_state_dict(checkpoint["net2"])
    net3.module.load_state_dict(checkpoint["net3"])

    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])

    saved_world_size = checkpoint["world_size"]

    if saved_world_size != world_size:
        raise RuntimeError(
            "Exact RNG-state restoration requires the same world size. "
            f"Checkpoint world size={saved_world_size}, "
            f"current world size={world_size}."
        )

    rng_states = checkpoint["rng_states"]

    restore_rng_state(rng_states[rank])

    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]

    if is_main_process:
        print(
            f"Resuming at epoch {start_epoch}, "
            f"global_step={global_step}"
        )

    return start_epoch, global_step


# ============================================================
# 10. RESUME
# ============================================================

start_epoch, global_step = load_checkpoint()


# ============================================================
# 11. TRAINING LOOP
# ============================================================

for epoch in range(start_epoch, EPOCHS):

    for model in models:
        model.train()

    # Change the distributed shuffle order each epoch.
    sampler.set_epoch(epoch)

    num_batches = len(loader)

    if num_batches == 0:
        raise RuntimeError(
            "The DataLoader has no complete batches. "
            "Check dataset size, world_size, and drop_last settings."
        )

    optimizer.zero_grad(set_to_none=True)

    epoch_loss_sum = torch.zeros(
        (),
        device=device,
        dtype=torch.float64,
    )

    for step, (x, target) in enumerate(
        make_batch_iterator(loader)
    ):

        x = x.to(
            device,
            non_blocking=(DEVICE == "cuda"),
        )

        target = target.to(
            device,
            non_blocking=(DEVICE == "cuda"),
        )

        # Handle a potentially shorter final accumulation group.
        group_start = (
            step // ACCUMULATION_STEPS
        ) * ACCUMULATION_STEPS

        group_size = min(
            ACCUMULATION_STEPS,
            num_batches - group_start,
        )

        is_last_microbatch = (
            (step + 1) % ACCUMULATION_STEPS == 0
            or step + 1 == num_batches
        )

        # All three DDP wrappers must suppress synchronization
        # on non-final micro-batches.
        with ExitStack() as stack:

            if not is_last_microbatch:
                for model in models:
                    stack.enter_context(model.no_sync())

            with torch.autocast(
                device_type=DEVICE,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):

                h1 = net1(x)
                h2 = net2(x)

                h = torch.cat([h1, h2], dim=1)

                output = net3(h)

                loss = nn.functional.mse_loss(
                    output,
                    target,
                )

                # Average gradients over the accumulation group.
                loss_for_backward = loss / group_size

            scaler.scale(loss_for_backward).backward()

        # Track the unscaled loss for logging.
        epoch_loss_sum += (
            loss.detach().to(torch.float64)
            * x.shape[0]
        )

        if is_last_microbatch:

            # Convert scaled gradients back to their actual scale.
            scaler.unscale_(optimizer)

            # Clip the synchronized, unscaled gradients.
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=MAX_GRAD_NORM,
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

            global_step += 1

    # One scheduler step per epoch.
    scheduler.step()

    # Calculate sample-weighted mean loss across all ranks.
    local_sample_count = torch.tensor(
        len(sampler),
        device=device,
        dtype=torch.float64,
    )

    dist.all_reduce(
        epoch_loss_sum,
        op=dist.ReduceOp.SUM,
    )

    dist.all_reduce(
        local_sample_count,
        op=dist.ReduceOp.SUM,
    )

    mean_epoch_loss = (
        epoch_loss_sum / local_sample_count
    )

    if is_main_process:
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {mean_epoch_loss.item():.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.7f} | "
            f"Global step: {global_step}"
        )

    # Collect each rank's RNG state before saving.
    local_rng_state = capture_rng_state()

    rank_rng_states = [None] * world_size

    dist.all_gather_object(
        rank_rng_states,
        local_rng_state,
    )

    # Save once per epoch, after all optimizer updates.
    save_checkpoint(
        epoch=epoch,
        global_step=global_step,
        rank_rng_states=rank_rng_states,
    )

    # Ensure all ranks finish the epoch's checkpoint stage
    # before proceeding.
    dist.barrier()


# ============================================================
# 12. CLEANUP
# ============================================================

dist.destroy_process_group()

if is_main_process:
    print("Training completed successfully!")

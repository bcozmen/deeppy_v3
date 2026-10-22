"""Single runtime configuration for execution and reproducibility."""

import os

import torch

from .amp import AMPConfig
from .compiler import Compiler
from .distributed import Distributed
from .rng import RandomSeed


class Runtime:
    """Initialize process runtime once and share it with models and optimizers.

    The runtime owns device placement, distributed wrapping, forward AMP
    autocasting, compilation, and process-wide RNG configuration. Models and
    optimizers receive only this object instead of repeating those arguments.
    """

    def __init__(
        self,
        device,
        *,
        amp=None,
        compile=False,
        compile_kwargs=None,
        seed=None,
        strict_determinism=False,
    ):
        self.distributed = Distributed(torch.device(device))
        self.rng = RandomSeed(seed, strict_determinism=strict_determinism)
        self.amp = AMPConfig(amp, device_type=self.device.type)
        compile_options = {} if compile is True else dict(compile_kwargs or {})
        self.compiler = Compiler(compile is True, **compile_options)

    @property
    def device(self):
        return self.distributed.device

    @property
    def rank(self):
        return self.distributed.rank

    @property
    def world_size(self):
        return self.distributed.world_size

    @property
    def is_main_process(self):
        return self.distributed.is_main_process

    def prepare_model(self, model):
        """Compile, move, and optionally DDP-wrap ``model``."""
        return self.distributed.wrap_model(self.compiler(model))

    def autocast(self):
        """Return the configured forward autocast context."""
        return self.amp.autocast()

    def save_rng_state(self, folder_path):
        """Save this process's RNG state in a rank-specific checkpoint file."""
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.rng.save_dict(), self._rng_path(folder_path))

    def load_rng_state(self, folder_path):
        """Restore this process's RNG state from a rank-specific file.

        Returns ``False`` when no RNG file exists, preserving the ability to
        load checkpoints created without RNG state.
        """
        path = self._rng_path(folder_path)
        if not os.path.exists(path):
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.rng.load_dict(state)
        return True

    def _rng_path(self, folder_path):
        return os.path.join(folder_path, f"rng_rank{self.rank}.pt")

    def barrier(self):
        self.distributed.barrier()

    def cleanup(self):
        self.distributed.cleanup()

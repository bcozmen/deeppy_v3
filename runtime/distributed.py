"""Device and optional distributed-training setup."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class Distributed:
	"""Configure the current process for single-device or DDP execution."""

	def __init__(self, device: torch.device):
		self.device = torch.device(device)
		self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if self.device.type == "cuda" else 0
		self.rank = 0
		self.world_size = 1
		self.is_distributed = False
		self._owns_process_group = False

		backend = self._configure_device()
		self._init_process_group(backend)

	def _configure_device(self) -> str:
		if self.device.type == "cuda":
			if not torch.cuda.is_available():
				raise RuntimeError("CUDA is not available.")
			if not 0 <= self.local_rank < torch.cuda.device_count():
				raise RuntimeError(
					f"LOCAL_RANK={self.local_rank}, but only "
					f"{torch.cuda.device_count()} GPUs are visible."
				)
			torch.cuda.set_device(self.local_rank)
			self.device = torch.device("cuda", self.local_rank)
			return "nccl"

		if self.device.type == "cpu":
			self.device = torch.device("cpu")
			return "gloo"

		raise ValueError(f"Unsupported device type: {self.device.type}")

	def _init_process_group(self, backend: str):
		self.rank = int(os.environ.get("RANK", 0))
		self.world_size = int(os.environ.get("WORLD_SIZE", 1))

		if self.world_size > 1 and not dist.is_initialized():
			dist.init_process_group(backend=backend)
			self._owns_process_group = True

		if dist.is_initialized():
			self.rank = dist.get_rank()
			self.world_size = dist.get_world_size()
		self.is_distributed = self.world_size > 1

	def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
		model = model.to(self.device)
		if not self.is_distributed:
			return model
		if self.device.type == "cuda":
			return DDP(model, device_ids=[self.device.index], output_device=self.device.index)
		return DDP(model)

	@property
	def is_main_process(self) -> bool:
		return self.rank == 0

	def barrier(self):
		if self.is_distributed:
			dist.barrier()

	def cleanup(self):
		if self._owns_process_group and dist.is_initialized():
			dist.destroy_process_group()


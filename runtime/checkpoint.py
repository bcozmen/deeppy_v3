"""Runtime-aware checkpointing."""

import os

import torch


class Checkpointer:
	"""Save and restore model state collectively across DDP ranks."""

	def __init__(self, folder_path, model, runtime, *, dataset=None):
		self.folder_path = folder_path
		self.checkpoint_file = os.path.join(folder_path, "checkpoint.pt")
		self.model = model
		self.runtime = runtime
		self.dataset = dataset
		os.makedirs(folder_path, exist_ok=True)

	def save(self):
		if self.runtime.is_main_process:
			state = {"model": self.model.save_states()}
			if self.dataset is not None and hasattr(self.dataset, "save_dict"):
				state["dataset"] = self.dataset.save_dict()
			torch.save(state, self.checkpoint_file)

		self.runtime.save_rng_state(self.folder_path)
		self.runtime.barrier()

	def load(self):
		self.runtime.barrier()
		state = torch.load(self.checkpoint_file, map_location="cpu", weights_only=False)
		self.model.load_states(state["model"])
		if self.dataset is not None and "dataset" in state:
			self.dataset.load_dict(state["dataset"])

		self.runtime.load_rng_state(self.folder_path)
		self.runtime.barrier()

__all__ = ["Checkpointer"]

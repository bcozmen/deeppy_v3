"""Random-state management for reproducible training."""
import random
import numpy as np
import torch

class Rng:
	"""Set, save, and restore Python, NumPy, and PyTorch RNG state."""

	def __init__(self, seed=None, strict_determinism=False):
		self.strict_determinism = strict_determinism
		self.seed = seed
		self._set_determinism(strict_determinism)
		if seed is not None:
			self._set_seed(seed)

	def _set_determinism(self, strict_determinism):
		self.strict_determinism = strict_determinism
		torch.backends.cudnn.deterministic = strict_determinism
		torch.backends.cudnn.benchmark = not strict_determinism
		torch.use_deterministic_algorithms(strict_determinism)

	def _set_seed(self, seed):
		np.random.seed(seed)
		random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)

	def save_dict(self):
		state = {
			"numpy": np.random.get_state(),
			"random": random.getstate(),
			"torch": torch.get_rng_state(),
			"strict_determinism": self.strict_determinism,
			"seed": self.seed,
		}
		if torch.cuda.is_available():
			state["torch_cuda"] = torch.cuda.get_rng_state_all()
		return state

	def load_dict(self, state):
		np.random.set_state(state["numpy"])
		random.setstate(state["random"])
		torch.set_rng_state(state["torch"])
		if torch.cuda.is_available() and "torch_cuda" in state:
			torch.cuda.set_rng_state_all(state["torch_cuda"])
		self.strict_determinism = state.get("strict_determinism", self.strict_determinism)
		self.seed = state.get("seed", self.seed)
		self._set_determinism(self.strict_determinism)



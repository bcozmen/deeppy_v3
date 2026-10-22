"""Runtime-aware model orchestration."""

from .base import BaseModel
from .example import ExampleModel

Model = ExampleModel

__all__ = ["BaseModel", "ExampleModel", "Model"]
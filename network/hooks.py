"""Reusable forward-hook utilities for inspecting network activations."""


class ActivationCapture:
    """Capture outputs from selected modules for the duration of a context.

    Args:
        modules: Mapping from a user-defined name to an ``nn.Module``.
        detach: Detach captured tensors from the autograd graph.
        cpu: Move detached captures to CPU.
    """

    def __init__(self, modules, *, detach=True, cpu=False):
        self.modules = dict(modules)
        self.detach = detach
        self.cpu = cpu
        self.values = {}
        self._handles = []

    def __enter__(self):
        self.values.clear()
        for name, module in self.modules.items():
            self._handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, name):
        def capture(_module, _inputs, output):
            value = output
            if self.detach and hasattr(value, "detach"):
                value = value.detach()
            if self.cpu and hasattr(value, "cpu"):
                value = value.cpu()
            self.values[name] = value

        return capture

    def get(self, name):
        """Return one captured value, raising ``KeyError`` if it was not run."""
        return self.values[name]

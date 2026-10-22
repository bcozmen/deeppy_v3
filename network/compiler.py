"""Optional model compilation."""


class ModelCompiler:
    """Apply ``torch.compile`` when explicitly enabled."""

    def __init__(self, enabled=False, **compile_kwargs):
        self.enabled = enabled
        self.compile_kwargs = compile_kwargs

    def __call__(self, model):
        if not self.enabled:
            return model

        import torch

        return torch.compile(model, **self.compile_kwargs)

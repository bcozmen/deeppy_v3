class Callback:
    def on_train_begin(self, trainer):
        pass

    def on_epoch_begin(self, trainer, epoch):
        pass

    def on_backward_end(self, trainer, loss):
        pass

    def on_before_optimizer_step(self, trainer):
        pass

    def on_optimizer_step_end(self, trainer, updated):
        pass

    def on_epoch_end(self, trainer, epoch):
        pass

    def on_train_end(self, trainer):
        pass

class CallbackManager:
    def __init__(self, callbacks=None):
        self.callbacks = list(callbacks or [])

    def emit(self, event, *args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, event, None)

            if method is not None:
                method(*args, **kwargs)



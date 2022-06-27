import tensorflow as tf

class Printlr(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        print(self.model.optimizer._decayed_lr(float).numpy())
        
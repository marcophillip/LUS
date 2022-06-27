import tensorflow as tf

class weighted_loss_fn(tf.keras.losses.Loss):
    def __init__(self,positive_weights,negative_weights):
        super(weighted_loss_fn,self).__init__()
        self.epsilon = tf.keras.backend.epsilon()
        self.positive_weights=positive_weights
        self.negative_weights=negative_weights
        
    def call(self,y_true,y_pred):  # ylogp +(1-y)log(1-p)
        y_pred = tf.cast(y_pred,tf.float32)
        y_true = tf.cast(y_true,tf.float32)
        loss = self.positive_weights*y_true*tf.math.log(y_pred+self.epsilon)+ self.negative_weights*(1-y_true)*tf.math.log(1-y_pred+self.epsilon)
        return -tf.reduce_mean(loss)
        
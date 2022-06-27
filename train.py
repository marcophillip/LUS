from data import *
import tensorflow as tf
from loss import weighted_loss_fn
from models import Eff
from callbacks_ import Printlr

if __name__ == '__main__':
    
    model_name="densenet121"
    
    checkpoint_filepath = "train_weights_combined_dataset_densenet2/"+model_name+"/cp-{epoch:04d}.ckpt"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_freq = 5*16,
    )

    tensorboard_cp=tf.keras.callbacks.TensorBoard(
        log_dir="train_weights_combined_dataset_densenet2/{}".format(model_name),
        update_freq=5*16,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',verbose=1, factor=0.01,
                              patience=5)
#     printlr = Printlr()
    negative_weights,positive_weights=weights()  
    loss = weighted_loss_fn(positive_weights,negative_weights)
    
            
    model = Eff(model_name)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0001,
        decay_steps=400,
        decay_rate=0.096,
        staircase=True)




    model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss = loss,
    metrics=['accuracy']
              )
    
    train_data = TrainGenerator().batch(16)
    validation_data = ValGenerator().batch(16)
    
    latest=tf.train.latest_checkpoint('train_weights_combined_dataset_densenet/densenet121')
    model.load_weights(latest)

    model.fit(train_data,
              validation_data=validation_data,
              epochs=500,
              callbacks=[
                       model_checkpoint_callback,
                         tensorboard_cp,
                         ]
                      )                      
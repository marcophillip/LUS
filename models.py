import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Dense,Dropout,MaxPooling2D,Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization,Activation,Add
from tensorflow.keras import activations
import numpy as np
from efficientnet import tfkeras as efficientnet



# import tensorflow as tf


# class SelfAttention(tf.keras.layers.Layer):
#     def __init__(self):
#         super(SelfAttention,self).__init__()
        
#     def build(self,input_shape):
# #         print(input_shape)
#         self.n,self.h_,self.w,self.c = input_shape
        
#         self.n_feats = self.h_*self.w
#         self.f = Conv2D(self.c//8, 1, padding='same')
        
#         self.g = Conv2D(self.c//8,1,padding='same')
#         self.g_pool = MaxPooling2D(2,2)
        
#         self.h = Conv2D(self.c//2,1,padding='same')
#         self.h_pool = MaxPooling2D(2,2)
        
#         self.attn = Conv2D(self.c,1,padding='same')
#         self.sigma = tf.Variable(0.0,dtype=tf.float32,trainable=True)
        
#     def call(self,x):
# #         print(x.shape)
        
#         f = self.f(x)
#         f = tf.reshape(f,[-1,self.n_feats,f.shape[-1]])
# #         print(f.shape)
        
#         g = self.g(x)
#         g = self.g_pool(g)
#         g = tf.reshape(g,[-1,self.n_feats//4,g.shape[-1]])
# #         print(g.shape)
        
        
#         attn = tf.matmul(f,g, transpose_b=True)
#         attn = tf.nn.softmax(attn)
# #         print(tf.reduce_sum(attn, axis=-1))
        
#         h = self.h(x)
#         h = self.h_pool(h)
# #         print(h.shape)
#         h = tf.reshape(h,[-1,self.n_feats//4,h.shape[-1]])
        
#         attn_2 = tf.matmul(attn, h)
# #         print(attn_2.shape)
        
#         attn_2 = tf.reshape(attn_2, [-1, self.h_, self.w, attn_2.shape[-1]])
#         attn_2 = self.attn(attn_2)
        
#         o = x+self.sigma*attn_2
#         return o





class Eff(tf.keras.models.Model):
    def __init__(self,model_name):
        super(Eff,self).__init__()
        self.model_name=model_name
        
       
        models = {
                    'VGG19':tf.keras.applications.VGG19(
                                      include_top=False,
#                                       pooling='avg',
                                      weights="imagenet",
                                      input_shape=(224,224, 3)),
                    'resnet50':tf.keras.applications.ResNet50(
                                      include_top=False,
#                                       pooling='avg',
                                      weights="imagenet",
                                      input_shape=(224,224, 3)),
            
                    'xception':tf.keras.applications.Xception(
                          include_top=False,
#                                       pooling='avg',
                          weights="imagenet",
                          input_shape=(224,224, 3)),
            
                    'densenet121':tf.keras.applications.DenseNet121(
                          include_top=False,
                          weights="imagenet",
                          input_shape=(224,224, 3)),
            
                    'efficientnetB0':efficientnet.EfficientNetB0(
                                      include_top=False,
#                                       pooling='avg',
                                      weights="noisy-student",
                                      input_shape=(224,224, 3)),

#                     'efficientnetB0':tf.keras.applications.EfficientNetB0(
#                                       include_top=False,
#                                       weights='imagenet',
#                                       input_shape=(224,224, 3)
#                                         ),
            
                    'efficientnetB1':tf.keras.applications.EfficientNetB1(
                                      include_top=False,
                                      weights="imagenet",
                                      input_shape=(240,240, 3)),

                    'efficientnetB2':tf.keras.applications.EfficientNetB2(
                                      include_top=False,
                                      weights="imagenet",
                                      input_shape=(260,260, 3)),

#                     'efficientnetB3':tf.keras.applications.EfficientNetB3(
#                                       include_top=False,
#                                       weights="imagenet",
#                                       input_shape=(300,300, 3)),
            
                    'efficientnetB3':efficientnet.EfficientNetB3(
                                      include_top=False,
#                                       pooling='avg',
                                      weights="noisy-student",
                                      input_shape=(300,300, 3)),
                    'efficientnetB4':tf.keras.applications.EfficientNetB4(
                                      include_top=False,
                                      weights="imagenet",
                                      input_shape=(380,380, 3)),
                }
        
        self.backbone = models[model_name]
        self.backbone.trainable=False
        
#         regularizer = tf.keras.regularizers.l2(0.01)

#         for layer in self.backbone.layers:
#             for attr in ['kernel_regularizer']:
#                 if hasattr(layer, attr):
#                     setattr(layer, attr, regularizer)
        
        for layer in self.backbone.layers[-9:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
                
        self.glb=keras.layers.GlobalAveragePooling2D()   
#         self.bn = keras.layers.BatchNormalization()
        self.dropout=keras.layers.Dropout(0.7)
#         self.dense1 = keras.layers.Dense(513,activation='relu')
#         self.dropout1=keras.layers.Dropout(0.2)
#         self.dense2 = keras.layers.Dense(1024,activation='relu')
#         self.dropout2=keras.layers.Dropout(0.2)

        self.out=keras.layers.Dense(4, activation='sigmoid')
        
    def call(self,inputs):
        x = self.backbone(inputs)
        x = self.glb(x)
#         x = self.bn(x)
        x = self.dropout(x)
#         x = self.dense1(x) 
#         x = self.dropout1(x)
#         x = self.dense2(x) 
#         x = self.dropout2(x)

        x = self.out(x)
        return x
      
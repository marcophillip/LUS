import tensorflow as tf
from skmultilearn.model_selection import iterative_train_test_split
import pandas as pd
import os
import cv2
import numpy as np
import math
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer

np.random.seed(42)
tf.random.set_seed(42)

local_data = pd.read_csv('inputs/cleaned_trainset2.csv')
# local_data = local_data.rename(columns={'Normal':'alines','>3 B-lines':'blines','Consolidation':'consolidation','Effusion':'effusion'})
# local_data.image=local_data.image.apply(lambda x:x.replace(' ','_'))
# local_data.image=local_data.image.apply(lambda x: 'dataset/denoised_data/'+x)

open_data = pd.read_csv('inputs/cleaned_opensource.csv')



# X_train, y_train, X_test, y_test = iterative_train_test_split(open_data.values, open_data[['alines','blines','consolidation','effusion']].values, test_size = 0.2)

# open_data['fold']=-1
# from skmultilearn.model_selection import IterativeStratification
# k_fold = IterativeStratification(n_splits=5, order=3)

# for i,(train, test) in  enumerate(k_fold.split(open_data.values, open_data[['alines','blines','consolidation','effusion']].values)):
#     open_data.loc[test,'fold'] =i

# train_df=pd.concat([
# open_data[open_data.folds==0],
# open_data[open_data.folds==1],
# open_data[open_data.folds==2],
# open_data[open_data.folds==3]
# ])

# val_df = open_data[open_data.folds==4]

train_df = local_data[local_data.group_kfold!=2]
val_df = local_data[local_data.group_kfold==2]

train_df=shuffle(pd.concat([train_df,open_data]))

def get_class_weights(dataframe):
    positive_weights=[]
    negative_weights=[]
    classes_data=dataframe[['alines','blines','consolidation','effusion']]
    class_dict={}
    total=len(classes_data)
    for c in classes_data.columns.tolist():
#         print(classes_data[c].value_counts().to_dict())
        dict_ = classes_data[c].value_counts().to_dict()
        neg = dict_[0]
        pos = dict_[1]
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_dict[c] = {0:weight_for_0,1:weight_for_1}
    
    for c in class_dict.keys():
        positive_weights.append(class_dict[c][1])
        negative_weights.append(class_dict[c][0])
    
    return np.array(negative_weights),np.array(positive_weights)

def weights():
    return get_class_weights(train_df)

train_transform = A.Compose([
#     A.CLAHE(clip_limit=(1,4), p= 1),
    A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0,scale_limit=(0.1,0.2),rotate_limit=0,border_mode=0,value=0,p=0.6),
    A.ShiftScaleRotate(shift_limit=0.0625,scale_limit=0,rotate_limit=10,border_mode=0,value=0)
])


class TrainGenerator(tf.data.Dataset):
    
    def generator():
        image_paths = train_df.image.values.tolist()
        classes = train_df[['alines', 'blines','consolidation','effusion']].values.tolist()
        
        i=0
        while i<len(train_df):
            try:
                image = cv2.imread(image_paths[i])
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

#                 transformed = train_transform(image=image)
#                 image=transformed['image']

                image= cv2.resize(image,(224,224))
         
                image = tf.keras.applications.densenet.preprocess_input(image)

                class_ = classes[i]

                yield image, class_

                i+=1
            except:
                i+=1
            
    def __new__(cls):
        return tf.data.Dataset.from_generator(cls.generator,
                                              output_shapes=((224,224,3),(4)),
                                              output_types=(tf.float32,tf.float32),
                                             )
            
            
class ValGenerator(tf.data.Dataset):
    
    def generator():
        image_paths =val_df.image.values.tolist()
        classes = val_df[['alines', 'blines','consolidation','effusion']].values.tolist()
        
        i=0
        while i<len(val_df):
            try:
                image = cv2.imread(image_paths[i])
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#                 image=A.CLAHE(clip_limit=(1,4), p= 1)(image=image)
                image = cv2.resize(image,(224,224))
         
                image = tf.keras.applications.densenet.preprocess_input(image)

                class_ = classes[i]

                yield image, class_
                i+=1
            except:
                i+=1

    def __new__(cls):
        return tf.data.Dataset.from_generator(cls.generator,
                                              output_shapes=((224,224,3),(4)),
                                              output_types=(tf.float32,tf.float32),
                                             )
        

        
        
test_df = pd.read_csv('inputs/cleaned_testset.csv')
test_df=test_df.drop([0,24,92,227,157,143,32,203,179,98,13,23,142,36,154,273,62,10,12,\
                      286,9,205,1,0,142,279,43,168,141,74,72,157]).reset_index()
# test_df=test_df.rename(columns={'Normal':'alines','>3 B-lines':'blines','Consolidation':'consolidation','Effusion':'effusion'})
# test_df.image=test_df.image.apply(lambda x:x.replace(' ','_'))
# test_df.image=test_df.image.apply(lambda x: 'dataset/denoised_data/'+x)

class TestGenerator(tf.data.Dataset):
    
    def generator():
        image_paths = test_df.image.values.tolist()
        classes = test_df[['alines', 'blines','consolidation','effusion']].values.tolist()
        
        i=0
        while i<len(test_df):
#             try:
            image = cv2.imread(image_paths[i])
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#             image=A.CLAHE(clip_limit=(1,4), p= 1)(image=image)

            image = cv2.resize(image,(224,224))

            image = tf.keras.applications.efficientnet.preprocess_input(image)

            class_ = classes[i]

            yield image, class_
            i+=1
#             except:
#                 i+=1

    def __new__(cls):
        return tf.data.Dataset.from_generator(cls.generator,
                                              output_shapes=((224,224,3),(4)),
                                              output_types=(tf.float32,tf.float32),
                                             )
        
        
        
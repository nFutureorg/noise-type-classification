# -*- coding: utf-8 -*-
"""noise-type-detection-in-images.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UQ79A5-VwPriBflTfzcH7VgP5_zM9gzN
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense,Flatten
from tensorflow.keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
# %matplotlib inline
from keras.models import Model
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Sequential
#from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD


#Dataset
train_path='dataset/train_im'
test_path='dataset/test_im'
#val_path='../input/bupcovidfunding/Lung Segmentation Data/Val'

img_height = 481
img_width = 321

train_batches= ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.1,
    horizontal_flip=True).flow_from_directory(train_path,subset='training',target_size=(img_height, img_width),batch_size=32,shuffle=True)
val_batches= ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.1,
    horizontal_flip=True).flow_from_directory(train_path,subset='validation', target_size=(img_height, img_width),batch_size=32,shuffle=False)
test_batches= ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True).flow_from_directory(test_path,target_size=(img_height, img_width),batch_size=32,shuffle=False)

#train_batches= ImageDataGenerator(preprocessing_function=keras.applications.densenet.preprocess_input).flow_from_directory(train_path,target_size=(224,224),batch_size=32,shuffle=True)
#val_batches= ImageDataGenerator(preprocessing_function=keras.applications.densenet.preprocess_input).flow_from_directory(val_path,target_size=(224,224),batch_size=32,shuffle=False)
#test_batches= ImageDataGenerator(preprocessing_function=keras.applications.densenet.preprocess_input).flow_from_directory(test_path,target_size=(224,224),batch_size=32,shuffle=False)

import tensorflow as tf
from keras.models import load_model
#Change model
model = tf.keras.applications.xception.Xception(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)
x = model.layers[-2].output
predictions = Dense(3,activation='softmax')(x)
model = Model(inputs=model.input,outputs=predictions)

model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

base = tf.keras.applications.xception.Xception(weights=None,
                                include_top=False,
                                input_shape=(img_height, img_width,3)
                               )


predictions = tf.keras.layers.Dense(14, activation='sigmoid', name='predictions')(base.output)
base = tf.keras.Model(inputs=base.input, outputs=predictions) 

new_model = tf.keras.layers.GlobalAveragePooling2D()(base.layers[-3].output) 
    ### OPT: add use flatten instead of global pooling. Opt: add dropout, fully connected layers after
new_model = tf.keras.layers.Dense(9, activation='softmax')(new_model) 
model = tf.keras.Model(base.input, new_model)

model.summary()

model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
checkpoint = ModelCheckpoint('results/xception_best.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')
his = model.fit(train_batches,steps_per_epoch=np.ceil(train_batches.samples / train_batches.batch_size),epochs =30 ,verbose=1,validation_data = val_batches,validation_steps=np.ceil(val_batches.samples / val_batches.batch_size),callbacks=[checkpoint,tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)])

#generate training curve
import matplotlib.pyplot as plt
import pandas as pd
history = his
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
df = pd.DataFrame(list(zip(train_acc, val_acc,train_loss,val_loss)),
               columns =['Training Acc', 'Validation Acc','Training Loss','Validation Loss'])
df.to_csv('curve.csv')
epochs = range(1,len(df['Validation Acc'])+1)
plt.plot(epochs, train_acc, 'g', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
#plt.title('Attention LSTM-CNN (FastText Embedding)')
plt.xlabel('Epochs', fontsize='medium')
plt.ylabel('Accuracy', fontsize='medium')
plt.legend()
#sn.set(font_scale=1)
plt.savefig('train.pdf', format='pdf', dpi=300)
plt.savefig('train.png', format='png', dpi=300)
plt.show()

#generate training curve
#import matplotlib.pyplot as plt
#history = his
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1,len(df['Validation Acc'])+1)
plt.plot(epochs, train_acc, 'g', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
#plt.title('Attention LSTM-CNN (FastText Embedding)')
plt.xlabel('Epochs', fontsize='medium')
plt.ylabel('Accuracy', fontsize='medium')
plt.legend()
#sn.set(font_scale=1)
plt.savefig('results/train.pdf', format='pdf', dpi=300)
plt.savefig('results/train.png', format='png', dpi=300)
plt.show()

# Commented out IPython magic to ensure Python compatibility.
#generate Result
from __future__ import print_function
import sklearn
from matplotlib import pyplot as plt 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
import h5py
import os
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

#Change HERE
model = load_model('results/xception_best.h5')


loss, acc = model.evaluate_generator(test_batches, steps=np.ceil(test_batches.samples / test_batches.batch_size), verbose=1)
print('accuracy:',acc)
predictions = model.predict_generator(test_batches, steps = np.ceil(test_batches.samples / test_batches.batch_size), verbose=1, workers=0) 
Y_pred = np.argmax(predictions, axis=1) 
print('Classification Report') 
print(classification_report(test_batches.classes, Y_pred))
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
original = test_batches.labels
cm=confusion_matrix(original,Y_pred)
print(cm)
y_true = ["0","1","2","3","4","5","6","7","8"]
#y_true=['Angry', 'Fear', 'Happy','Neutral','Sad']
data = cm
class1_acc = data[0][0]/(data[0][0]+data[0][1]+data[0][2] + data[0][3]+data[0][4]+data[0][5] + data[0][6]+data[0][7]+data[0][8])

class2_acc = data[1][1]/(data[1][0]+data[1][1]+data[1][2] + data[1][3]+data[1][4]+data[1][5] + data[1][6]+data[1][7]+data[1][8])

class3_acc = data[2][2]/(data[2][0]+data[2][1]+data[2][2] + data[2][3]+data[2][4]+data[2][5] + data[2][6]+data[2][7]+data[2][8])

class4_acc = data[3][3]/(data[3][0]+data[3][1]+data[3][2] + data[3][3]+data[3][4]+data[3][5] + data[3][6]+data[3][7]+data[3][8])

class5_acc = data[4][4]/(data[4][0]+data[4][1]+data[4][2] + data[4][3]+data[4][4]+data[4][5] + data[4][6]+data[4][7]+data[4][8])

class6_acc = data[5][5]/(data[5][0]+data[5][1]+data[5][2] + data[5][3]+data[5][4]+data[5][5] + data[5][6]+data[5][7]+data[5][8])

class7_acc = data[6][6]/(data[6][0]+data[6][1]+data[6][2] + data[6][3]+data[6][4]+data[6][5] + data[6][6]+data[6][7]+data[6][8])

class8_acc = data[7][7]/(data[7][0]+data[7][1]+data[7][2] + data[7][3]+data[7][4]+data[7][5] + data[7][6]+data[7][7]+data[7][8])

class9_acc = data[8][8]/(data[8][0]+data[8][1]+data[8][2] + data[8][3]+data[8][4]+data[8][5] + data[8][6]+data[8][7]+data[8][8])

print('Erlang acc: ',class1_acc)
print('Exponential acc: ',class2_acc)
print('Gaussian acc: ',class3_acc)

print('Lognormal acc: ',class4_acc)
print('Poisson acc: ',class5_acc)
print('Rayleigh acc: ',class6_acc)

print('Salt and Pepper acc: ',class7_acc)
print('Speckle acc: ',class8_acc)
print('Uniform acc: ',class9_acc)

df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
#df_cm.index.name = 'Actual'
#df_cm.columns.name = 'Predicted'
sn.set(font_scale=2)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 24}, fmt="d")
plt.savefig('results/Cm_test.pdf', format='pdf', dpi=300)
plt.savefig('results/Cm_test.png', format='png', dpi=300)

# Commented out IPython magic to ensure Python compatibility.
#generate Result
from __future__ import print_function
import sklearn
from matplotlib import pyplot as plt 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
import h5py
import os
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

#Change HERE
model = load_model('results/xception_best.h5')


loss, acc = model.evaluate_generator(val_batches, steps=np.ceil(val_batches.samples / val_batches.batch_size), verbose=1)
print('accuracy:',acc)
predictions = model.predict_generator(val_batches, steps = np.ceil(val_batches.samples / val_batches.batch_size), verbose=1, workers=0) 
Y_pred = np.argmax(predictions, axis=1) 
print('Classification Report') 
print(classification_report(val_batches.classes, Y_pred))
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
original = val_batches.labels
cm=confusion_matrix(original,Y_pred)
print(cm)
y_true = ["0","1","2","3","4","5","6","7","8"]
#y_true=['Angry', 'Fear', 'Happy','Neutral','Sad']
data = cm
class1_acc = data[0][0]/(data[0][0]+data[0][1]+data[0][2] + data[0][3]+data[0][4]+data[0][5] + data[0][6]+data[0][7]+data[0][8])

class2_acc = data[1][1]/(data[1][0]+data[1][1]+data[1][2] + data[1][3]+data[1][4]+data[1][5] + data[1][6]+data[1][7]+data[1][8])

class3_acc = data[2][2]/(data[2][0]+data[2][1]+data[2][2] + data[2][3]+data[2][4]+data[2][5] + data[2][6]+data[2][7]+data[2][8])

class4_acc = data[3][3]/(data[3][0]+data[3][1]+data[3][2] + data[3][3]+data[3][4]+data[3][5] + data[3][6]+data[3][7]+data[3][8])

class5_acc = data[4][4]/(data[4][0]+data[4][1]+data[4][2] + data[4][3]+data[4][4]+data[4][5] + data[4][6]+data[4][7]+data[4][8])

class6_acc = data[5][5]/(data[5][0]+data[5][1]+data[5][2] + data[5][3]+data[5][4]+data[5][5] + data[5][6]+data[5][7]+data[5][8])

class7_acc = data[6][6]/(data[6][0]+data[6][1]+data[6][2] + data[6][3]+data[6][4]+data[6][5] + data[6][6]+data[6][7]+data[6][8])

class8_acc = data[7][7]/(data[7][0]+data[7][1]+data[7][2] + data[7][3]+data[7][4]+data[7][5] + data[7][6]+data[7][7]+data[7][8])

class9_acc = data[8][8]/(data[8][0]+data[8][1]+data[8][2] + data[8][3]+data[8][4]+data[8][5] + data[8][6]+data[8][7]+data[8][8])

print('Erlang acc: ',class1_acc)
print('Exponential acc: ',class2_acc)
print('Gaussian acc: ',class3_acc)

print('Lognormal acc: ',class4_acc)
print('Poisson acc: ',class5_acc)
print('Rayleigh acc: ',class6_acc)

print('Salt and Pepper acc: ',class7_acc)
print('Speckle acc: ',class8_acc)
print('Uniform acc: ',class9_acc)

df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
#df_cm.index.name = 'Actual'
#df_cm.columns.name = 'Predicted'
sn.set(font_scale=2)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 24}, fmt="d")
plt.savefig('results/Cm_val.pdf', format='pdf', dpi=300)
plt.savefig('results/Cm_val.png', format='png', dpi=300)
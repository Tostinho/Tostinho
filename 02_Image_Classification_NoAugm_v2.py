#--------------------------------------------------------------------------------#
#
#                           CLASSIFICAZIONE DELLE IMMAGINI
#
#--------------------------------------------------------------------------------#

#https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/



#### ----------------------------- ####
####     IMPORT DEI PACCHETTI      ####
#### ----------------------------- ####

import matplotlib.pyplot as plt
import seaborn as sns

import keras
import tensorflow as tf


from tensorflow.keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , BatchNormalization, Flatten , Dropout 

from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report,confusion_matrix

import cv2
import os

import numpy as np
import pandas as pd




#### --------------------------------------------- ####
####  CREAZIONE DELLE CARTELLE DI TRAINING E TEST  ####
#### --------------------------------------------- ####

#pip install split-folders
#pip install split-folders tqdm

import splitfolders

### Split
splitfolders.ratio("C:/Users/angel/OneDrive/Desktop/MEDIA/O.R.1.3/Classificazione_MRI/CLAHE_PNG", 
                   output= "TRAIN_VAL",
                   seed=1337, 
                   ratio=(.8, .2), 
                   group_prefix=None
                  )



#### -------------------------------------------- ####
####  CARICAMENTO DATI ( con resize a 224 pixel)  ####
#### -------------------------------------------- ####

labels = ['01_NoArt_CLAHE', '02_Motion_CLAHE', '03_Metal_CLAHE', '04_Altro_CLAHE']
img_size = 224

def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir,label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data,dtype=object) #ho dovuto specificare il dtype



train = get_data('C:/Users/angel/OneDrive/Desktop/MEDIA/O.R.1.3/Classificazione_MRI/TRAIN_VAL/train')
val = get_data('C:/Users/angel/OneDrive/Desktop/MEDIA/Classificazione_MRI/TRAIN_VAL/val')


### Visualizzazione dei dati

## Train
l = []
for i in train:
    if(i[1] == 0):
        l.append("01_NoArt")
    if(i[1] == 1):
        l.append("02_Motion")
    if(i[1] == 2):
        l.append("03_Metal")
    if(i[1] == 3):
        l.append("04_Altro")
        
l=pd.DataFrame(l)
# Count Plot
sns.set_style('darkgrid')
sns.countplot(data=l,x=0)


# Le prime due MRI
plt.figure(figsize = (5,5)) #01_NoArt
plt.imshow(train[1][0])             ## Capire come si prendono le diverse classi
plt.title(labels[train[1][1]])

plt.figure(figsize = (5,5)) #02_Motion
plt.imshow(train[110][0])
plt.title(labels[train[110][1]])

plt.figure(figsize = (5,5)) #03_Metal
plt.imshow(train[190][0])
plt.title(labels[train[190][1]])

plt.figure(figsize = (5,5)) #04_Altro
plt.imshow(train[-1][0])
plt.title(labels[train[-1][1]])



## Validation
l = []
for i in val:
    if(i[1] == 0):
        l.append("01_NoArt")
    if(i[1] == 1):
        l.append("02_Motion")
    if(i[1] == 2):
        l.append("03_Metal")
    if(i[1] == 3):
        l.append("04_Altro")
        
# Count Plot

l=pd.DataFrame(l)
# Count Plot
sns.set_style('darkgrid')
sns.countplot(data=l,x=0)






#### ---------------------------------------- ####
####    DATA PRE PROCESSING  ####
#### ---------------------------------------- ####



### Processing
x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)


# Normalizzazione delle immagini dividendo per il loro valore massimo
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)

y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)

y_val = np.array(y_val)

### Data Augmentation

#Data augmentation is a strategy used to increase the amount of data by using techniques like cropping, padding, flipping, etc.
#Data augmentation makes the model more robust to slight variations, and hence prevents the model from overfitting.
#https://towardsdatascience.com/exploring-image-data-augmentation-with-keras-and-tensorflow-a8162d89b844

#datagen = ImageDataGenerator(
#        featurewise_center=False,  # set input mean to 0 over the dataset
#        samplewise_center=False,  # set each sample mean to 0
#        featurewise_std_normalization=False,  # divide inputs by std of the dataset
#        samplewise_std_normalization=False,  # divide each input by its std
#        zca_whitening=False,  # apply ZCA whitening
#        rotation_range = False,  # randomly rotate images in the range (degrees, 0 to 180)
#        zoom_range = False, # Randomly zoom image 
#       width_shift_range= 0.1,  # randomly shift images horizontally (fraction of total width)
#        height_shift_range= 0.1,  # randomly shift images vertically (fraction of total height)
#        horizontal_flip = True,  # randomly flip images
#        vertical_flip=False)  # randomly flip images

# Creazione dei dati a partire dal Training set
#datagen.fit(x_train)
    


#### ---------------------------------- ####
####        MODEL IMPLEMENTATION        ####
#### ---------------------------------- ####


####    MODEL 1    ####
model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
model.add(MaxPool2D())
model.add(Conv2D(32,3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Conv2D(64,3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(4, activation="softmax"))
model.summary()


opt = keras.optimizers.Adam(learning_rate= 0.000001)

model.compile(optimizer = opt,loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])

history = model.fit(x_train,y_train,epochs = 100, validation_data = (x_val, y_val))
max(history.history['accuracy'])
#0.9411


### TESTING

## Plot
acc = history.history['accuracy']; val_acc = history.history['val_accuracy']
loss = history.history['loss']; val_loss = history.history['val_loss']

epochs_range = range(100)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

## Prediction
predictions = model.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = ['NoArt (Class 0)','Motion (Class 1)','Metal (Class 2)','Other (Class 3)']))

#                    precision    recall  f1-score   support

#  NoArt (Class 0)       1.00      0.44      0.61        16
#Motion (Class 1)       0.51      0.88      0.65        26
# Metal (Class 2)       1.00      0.43      0.60        14
# Other (Class 3)       0.76      0.70      0.73        23

#        accuracy                           0.66        79
#       macro avg       0.82      0.61      0.65        79
#    weighted avg       0.77      0.66      0.65        79


### SALVATAGGIO MODELLO

model.save_weights ('model_1_noAug_whg.h5') 
model.save ('model_1_noAug')

### CARICAMENTO MODELLO
from tensorflow import keras
model_x = keras.models.load_model('model_1')

## Prediction
#predictions = model_x.predict_classes(x_val)
#predictions = predictions.reshape(1,-1)[0]
#print(classification_report(y_val, predictions, target_names = ['NoArt (Class 0)','Motion (Class 1)','Metal (Class 2)','Other (Class 3)']))




####    MODEL 2    ####

model2 = Sequential()
model2.add(Conv2D(32,3, padding="same", activation="relu", input_shape=(224,224,3)))
model2.add(MaxPool2D())
model2.add(Conv2D(32,3, padding="same", activation="relu"))
model2.add(MaxPool2D())
model2.add(Conv2D(64,3, padding="same", activation="relu"))
model2.add(MaxPool2D())
#model2.add(Dropout(0.2))
model2.add(Flatten())
model2.add(Dense(128,activation="relu"))
model2.add(Dense(4, activation="softmax"))
model2.summary()

opt = keras.optimizers.Adam(learning_rate= 0.000001)

model2.compile(optimizer = opt,loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])

history = model2.fit(x_train,y_train,epochs = 150, validation_data = (x_val, y_val))
max(history.history['accuracy'])
#0.9967


### TESTING

## Plot
acc = history.history['accuracy']; val_acc = history.history['val_accuracy']
loss = history.history['loss']; val_loss = history.history['val_loss']

epochs_range = range(150)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

## Prediction
predictions = model2.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = ['NoArt (Class 0)','Motion (Class 1)','Metal (Class 2)','Other (Class 3)']))

#                    precision    recall  f1-score   support

# NoArt (Class 0)       0.80      0.75      0.77        16
#Motion (Class 1)       0.68      0.73      0.70        26
# Metal (Class 2)       0.93      0.93      0.93        14
# Other (Class 3)       0.73      0.70      0.71        23

#        accuracy                           0.76        79
#       macro avg       0.78      0.78      0.78        79
#    weighted avg       0.76      0.76      0.76        79


### SALVATAGGIO MODELLO

model2.save_weights ('04_Modelli/model_2_noAug_whg.h5') 
model2.save ('04_Modelli/model_2_noAug')

### CARICAMENTO MODELLO
from tensorflow import keras
model2 = keras.models.load_model('04_Modelli/model_2_noAug')




####    MODEL 3    ####

model3 = Sequential()
model3.add(Conv2D(32, 6, padding="same", activation="relu", input_shape=(224,224,3)))
model3.add(MaxPool2D((2,2)))
model3.add(Conv2D(64, 6, padding="same", activation="relu"))
model3.add(MaxPool2D((2,2)))
model3.add(Conv2D(64, 3, padding="same", activation="relu"))
model3.add(MaxPool2D((2,2)))
#model2.add(Dropout(0.2))
model3.add(Flatten())
model3.add(Dense(64,activation="relu"))
model3.add(Dense(4, activation="softmax"))
#model.summary()

opt = adam(lr=0.00001)

model3.compile(optimizer = opt,loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])

history = model3.fit(x_train,y_train,epochs = 100, validation_data = (x_val, y_val))
max(history.history['accuracy'])
#0.9150


### TESTING

## Plot
acc = history.history['accuracy']; val_acc = history.history['val_accuracy']
loss = history.history['loss']; val_loss = history.history['val_loss']

epochs_range = range(100)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

## Prediction
predictions = model3.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = ['NoArt (Class 0)','Motion (Class 1)','Metal (Class 2)','Other (Class 3)']))

#                    precision    recall  f1-score   support

# NoArt (Class 0)       0.88      0.44      0.58        16
#Motion (Class 1)       0.52      0.85      0.65        26
# Metal (Class 2)       0.92      0.79      0.85        14
# Other (Class 3)       0.76      0.57      0.65        23

#        accuracy                           0.67        79
#       macro avg       0.77      0.66      0.68        79
#    weighted avg       0.73      0.67      0.67        79


### SALVATAGGIO MODELLO

model3.save_weights ('04_Modelli/model_3_noAug_whg.h5') 
model3.save ('04_Modelli/model_3_noAug')

### CARICAMENTO MODELLO
from tensorflow import keras
model_x = keras.models.load_model('model_3')




####    MODEL 4    ####

model4 = Sequential()
model4.add(Conv2D(32, 6, padding="same", activation="relu", input_shape=(224,224,3)))
model4.add(MaxPool2D((2,2)))
model4.add(Conv2D(64, 6, padding="same", activation="relu"))
model4.add(MaxPool2D((2,2)))
model4.add(Conv2D(64, 3, padding="same", activation="relu"))
model4.add(MaxPool2D((2,2)))
#model2.add(Dropout(0.2))
model4.add(Flatten())
model4.add(Dense(128,activation="relu"))
model4.add(Dense(64,activation="relu"))
model4.add(Dense(4, activation="softmax"))
#model.summary()

opt = adam(lr=0.00001)

model4.compile(optimizer = opt,loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])

history= model4.fit(x_train,y_train,epochs = 150, validation_data = (x_val, y_val))
max(history.history['accuracy'])
#1


### TESTING

## Plot
acc = history.history['accuracy']; val_acc = history.history['val_accuracy']
loss = history.history['loss']; val_loss = history.history['val_loss']

epochs_range = range(150)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

## Prediction
predictions = model4.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = ['NoArt (Class 0)','Motion (Class 1)','Metal (Class 2)','Other (Class 3)']))

#                    precision    recall  f1-score   support

# NoArt (Class 0)       0.86      0.75      0.80        16
#Motion (Class 1)       0.69      0.69      0.69        26
# Metal (Class 2)       1.00      0.86      0.92        14
# Other (Class 3)       0.74      0.87      0.80        23

#        accuracy                           0.78        79
#       macro avg       0.82      0.79      0.80        79
#    weighted avg       0.79      0.78      0.79        79


### SALVATAGGIO MODELLO

model4.save_weights ('04_Modelli/model_4_noAug_whg.h5') 
model4.save ('04_Modelli/model_4_noAug')

### CARICAMENTO MODELLO
from tensorflow import keras
model_x = keras.models.load_model('model_4')




####    MODEL 5    ####

model5 = Sequential()
model5.add(Conv2D(32, 3, padding="same", activation="elu", input_shape=(224,224,3)))
model5.add(MaxPool2D((2,2)))
model5.add(Conv2D(64, 3, padding="same", activation="elu"))
model5.add(MaxPool2D((2,2)))
model5.add(Conv2D(64, 3, padding="same", activation="elu"))
model5.add(MaxPool2D((2,2)))
model5.add(Dropout(0.2))
model5.add(Flatten())
model5.add(Dense(128,activation="elu"))
model5.add(Dense(64,activation="elu"))
model5.add(Dense(4, activation="softmax"))
#model.summary()

opt = adam(lr=0.0001)

model5.compile(optimizer = opt,loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])

history= model5.fit(x_train,y_train,epochs = 150, validation_data = (x_val, y_val))
max(history.history['accuracy'])
#1

### TESTING

## Plot
acc = history.history['accuracy']; val_acc = history.history['val_accuracy']
loss = history.history['loss']; val_loss = history.history['val_loss']

epochs_range = range(150)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

## Prediction
predictions = model5.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = ['NoArt (Class 0)','Motion (Class 1)','Metal (Class 2)','Other (Class 3)']))

#                    precision    recall  f1-score   support

# NoArt (Class 0)       0.92      0.75      0.83        16
#Motion (Class 1)       0.76      0.73      0.75        26
# Metal (Class 2)       0.78      1.00      0.88        14
# Other (Class 3)       0.78      0.78      0.78        23

#        accuracy                           0.80        79
#       macro avg       0.81      0.82      0.81        79
#    weighted avg       0.80      0.80      0.80        79


### SALVATAGGIO MODELLO

model5.save_weights ('04_Modelli/model_5_noAug_whg.h5') 
model5.save ('04_Modelli/model_5_noAug')

### CARICAMENTO MODELLO
from tensorflow import keras
model_x = keras.models.load_model('model_5')




####    MODEL 6 CLAHE   ####

model6 = Sequential()
model6.add(Conv2D(32, 3, padding="same", activation="elu", input_shape=(224,224,3)))
model6.add(BatchNormalization())
model6.add(MaxPool2D((2,2)))
model6.add(Conv2D(64, 3, padding="same", activation="elu"))
model6.add(BatchNormalization())
model6.add(MaxPool2D((2,2)))
model6.add(Conv2D(64, 3, padding="same", activation="elu"))
model6.add(BatchNormalization())
model6.add(MaxPool2D((2,2)))
model6.add(Dropout(0.2))
model6.add(Flatten())
model6.add(Dense(128,activation="elu"))
model6.add(Dense(64,activation="elu"))
model6.add(Dense(4, activation="softmax"))
#model.summary()

opt = adam(lr=0.0001)

model6.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])

history= model6.fit(x_train,y_train,epochs = 150, validation_data = (x_val, y_val))
max(history.history['accuracy'])
#1

### TESTING

## Plot
acc = history.history['accuracy']; val_acc = history.history['val_accuracy']
loss = history.history['loss']; val_loss = history.history['val_loss']

epochs_range = range(150)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

## Prediction
predictions = model6.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = ['NoArt (Class 0)','Motion (Class 1)','Metal (Class 2)','Other (Class 3)']))

#                    precision    recall  f1-score   support

# NoArt (Class 0)       0.92      0.75      0.83        16
#Motion (Class 1)       0.73      0.92      0.81        26
# Metal (Class 2)       0.93      1.00      0.97        14
# Other (Class 3)       0.89      0.70      0.78        23

#        accuracy                           0.84        79
#       macro avg       0.87      0.84      0.85        79
#    weighted avg       0.85      0.84      0.83        79

test_loss, test_acc = model6.evaluate(x_val,  y_val, verbose=2)
#loss: 0.5715 - accuracy: 0.8354
print('\nTest accuracy:', test_acc)
#Test accuracy: 0.83544


### SALVATAGGIO MODELLO

model6.save_weights ('04_Modelli/model_6_noAug_clahe_whg.h5') 
model6.save ('04_Modelli/model_6_noAug_clahe')

### CARICAMENTO MODELLO
from tensorflow import keras
model6 = keras.models.load_model('04_Modelli/model_6_noAug_clahe')




####    MODEL 7  CLAHE   ####

model7 = Sequential()
model7.add(Conv2D(32, 3, padding="same", activation="elu", input_shape=(224,224,3)))
model7.add(BatchNormalization())
model7.add(MaxPool2D((2,2)))
model7.add(Conv2D(64, 3, padding="same", activation="elu"))
model7.add(BatchNormalization())
model7.add(MaxPool2D((2,2)))
model7.add(Conv2D(64, 3, padding="same", activation="elu"))
model7.add(BatchNormalization())
model7.add(MaxPool2D((2,2)))
model7.add(Dropout(0.2))
model7.add(Flatten())
model7.add(Dense(128,activation="elu"))
model7.add(Dense(64,activation="elu"))
model7.add(Dense(4, activation="softmax"))
#model.summary()

opt = adam(lr=0.0001)

model7.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])

history= model7.fit(x_train,y_train,epochs =40, validation_data = (x_val, y_val))
max(history.history['accuracy'])
#1

### TESTING

## Plot
acc = history.history['accuracy']; val_acc = history.history['val_accuracy']
loss = history.history['loss']; val_loss = history.history['val_loss']

epochs_range = range(40)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

## Prediction
predictions = model7.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = ['NoArt (Class 0)','Motion (Class 1)','Metal (Class 2)','Other (Class 3)']))

#                    precision    recall  f1-score   support

# NoArt (Class 0)       0.80      0.75      0.77        16
#Motion (Class 1)       0.52      0.96      0.68        26
# Metal (Class 2)       0.92      0.79      0.85        14
# Other (Class 3)       1.00      0.17      0.30        23

#        accuracy                           0.66        79
#       macro avg       0.81      0.67      0.65        79
#    weighted avg       0.79      0.66      0.62        79




#### --------------------------------- ####
####      OTHER PRE-TRAINED MODEL      ####
#### --------------------------------- ####



####    MobileNetV2    ####

### Iportazione del modello
base_model = tf.keras.applications.MobileNetV2(input_shape = (224, 224, 3), include_top = False, weights = "imagenet")
base_model.trainable = False

MobileNetV2_model = tf.keras.Sequential([base_model,
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 #tf.keras.layers.Dropout(0.2),
                                 tf.keras.layers.Dense(128, activation="softmax"),
                                 tf.keras.layers.Dense(4, activation="softmax")                                     
                                ])

base_learning_rate = 0.0001
MobileNetV2_model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = MobileNetV2_model.fit(x_train,y_train,epochs = 150, validation_data = (x_val, y_val))


### TESTING
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(150)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

predictions_tl = MobileNetV2_model.predict_classes(x_val)
predictions_tl = predictions_tl.reshape(1,-1)[0]

print(classification_report(y_val, predictions_tl, target_names = ['NOART (Class 0)','MOVIMENTO (Class 1)','METALLO (Class 2)','ALTRO (Class 3)']))
#                      precision    recall  f1-score   support
 
#    NOART (Class 0)       0.88      0.44      0.58        16
#MOVIMENTO (Class 1)       0.63      0.85      0.72        26
#  METALLO (Class 2)       0.93      0.93      0.93        14
#    ALTRO (Class 3)       0.82      0.78      0.80        23

#           accuracy                           0.76        79
#          macro avg       0.81      0.75      0.76        79
#       weighted avg       0.79      0.76      0.75        79



### SALVATAGGIO MODELLO

MobileNetV2_model.save_weights ('04_Modelli/MobileNetV2_model_1_noAug_whg.h5') 
MobileNetV2_model.save ('04_Modelli/MobileNetV2_model_noAug_1')

### CARICAMENTO MODELLO
#from tensorflow import keras
#model_x = keras.models.load_model('MobileNetV2_model_1')







#TRANSFER LEARNING 1, potrebbe essere un'ottima soluzione ma la memoria della macchina non è suff
#importazione del modello

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()
        if p_1 > p:
            return input_img
        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
            if left + w <= img_w and top + h <= img_h:
                break
        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c
        return input_img
    return eraser


#imporazione library
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, UpSampling2D, Flatten, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from keras.datasets import cifar100
import tensorflow as tf
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.transform import resize
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator


num_classes = 4
nb_epochs = 50

#(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_test=x_val
y_test=y_val
#Pre-process the data
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

datagen = ImageDataGenerator(preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=True))
datagen.fit(x_train)


y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in resnet_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False

model = Sequential()
model.add(UpSampling2D())
model.add(UpSampling2D())
model.add(UpSampling2D())
model.add(resnet_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(.25))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

t=time.time()
historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                  batch_size=64),
                                  steps_per_epoch=x_train.shape[0] // 64,
                                  epochs=15,
                                  validation_data=(x_test, y_test))
print('Training time: %s' % (t - time.time()))
model.summary()










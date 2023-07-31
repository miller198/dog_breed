import numpy as np
import pandas as pd
import os, random, time, shutil, csv
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tqdm as tqdm
np.random.seed(42)
# %matplotlib inline

import json
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tqdm import tqdm
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras import optimizers


# Read train labels
labels_df = pd.read_csv('./datasets/labels.csv')

# Create list of alphabetically sorted labels
dog_breeds = sorted(list(set(labels_df['breed'])))

#  # of dog breeds
n_classes = len(dog_breeds)  

# Map each label string to an integer label
class_to_num = dict(zip(dog_breeds, range(n_classes)))

# Define image path
train_dir = './datasets/images/train/'
valid_dir = './datasets/images/valid/'
test_dir = './datasets/images/test/'


img_path = os.path.join(train_dir,np.random.choice(os.listdir(train_dir)))


# 모든 이미지를 1/255로 스케일을 조정 - 기존 방식, 과적합 발생
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # 타깃 디렉터리
        train_dir,
        # 모든 이미지를 150 × 150 크기로
        target_size=(150, 150),
        batch_size=20,
        # binary_crossentropy 손실을 사용하기 때문에 이진 레이블이 필요하다
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        valid_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')


for data_batch, labels_batch in train_generator:
    print('배치 데이터 크기:', data_batch.shape)
    print('배치 레이블 크기:', labels_batch.shape)
    break

# Create new model
model = Model(inputs=base_model.input, outputs=predictions)
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


x_train = []
y_train = []

for img_path in labels_df['id'].values:
    img_path = os.path.join(train_dir,img_path+'.jpg')
    img = load_img(img_path, target_size=(128, 128, 1))
    x = np.array(img).astype("float32")
    x = preprocess_input(x)
    x_train.append(x)
    y_train.append(class_to_num[labels_df[labels_df['id'] == img_path.split('/')[-1].split('.')[0]]['breed'].values[0]])

X_train = np.array(x_train)
y_train = to_categorical(y_train, num_classes=n_classes)
X_train, y_train = shuffle(X_train, y_train)
# Number of epochs to train
num_epochs = 50
# Fit the model
history = model.fit(X_train, y_train, validation_data = val, epochs=num_epochs, batch_size=64)



model.save('dog_breed_1.h5')


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# # Plot the accuracy and loss curves
# plt.plot(history.history['acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train'], loc='upper left')
# plt.show()
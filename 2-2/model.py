import csv
import cv2
import numpy as np
import datetime
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
lines = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

def crop_and_resize_change_color_space(image):
    image = cv2.cvtColor(cv2.resize(image[60:140,:], (32,32)), cv2.COLOR_BGR2RGB)
    return image

def generator_images(data, batchSize = 32):
    while True:
        data = shuffle(data)
        for i in range(0, len(data), int(batchSize/4)):
            X_batch = []
            y_batch = []
            details = data[i: i+int(batchSize/4)]
            for line in details:
                image = crop_and_resize_change_color_space(cv2.imread('./data/IMG/'+ line[0].split('/')[-1]))
                steering_angle = float(line[3])
                X_batch.append(image)
                y_batch.append(steering_angle)
                X_batch.append(np.fliplr(image))
                y_batch.append(-steering_angle)
                X_batch.append(crop_and_resize_change_color_space(cv2.imread('./data/IMG/'+ line[1].split('/')[-1])))
                y_batch.append(steering_angle+0.2)
                X_batch.append(crop_and_resize_change_color_space(cv2.imread('./data/IMG/'+ line[2].split('/')[-1])))
                y_batch.append(steering_angle-0.2)
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield shuffle(X_batch, y_batch)

training_data, validatio_data = train_test_split(lines, test_size = 0.2)

from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, MaxPooling2D, Flatten, Activation, Dense, Cropping2D, Lambda

model = Sequential()
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(32,32,3) ))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation = 'relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(80))
model.add(Dense(1))

model.compile('adam', 'mse')
model.fit_generator(generator_images(training_data), samples_per_epoch = len(training_data)*4, nb_epoch = 2, validation_data=generator_images(validatio_data), nb_val_samples=len(validatio_data))

model.save('model.h5')
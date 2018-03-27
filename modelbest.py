import csv
import cv2
import numpy as np
lines = []

def generator_images(lines, batchSize = 32):
    while True:
        X_batch = []
        y_batch = []
        for i in range(0, len(lines), int(batchSize/4)):
            line = lines[i: i+int(batchSize/4)]
            images = []
            measurements = []
            for l in line:
                image = cv2.imread('./data/IMG/'+ l[0].split('/')[-1])
                img_left = cv2.imread('./data/IMG/'+ l[1].split('/')[-1])
                image = cv2.resize(image, (64, 64))
                img_left = cv2.resize(img_left, (64, 64))
                image_flipped = np.fliplr(image)
                img_left_flipped = np.fliplr(img_left)
                images.extend([image, image_flipped, img_left, img_left_flipped])
                measurement = float(l[3])
                measurement_left = measurement+0.25
                measurement_flipped = -measurement
                measurement_left_flipped = -measurement_left
                measurements.extend([measurement, measurement_flipped, measurement_left, measurement_left_flipped])
            X_batch = np.array(images)
            y_batch = np.array(measurements)
            yield X_batch, y_batch

def generator_valid(lines, batchSize = 32):
    while True:
        X_valid = []
        y_valid = []
        images = []
        measurements = []
        for i in range(0, batchSize):
            rand = int(np.random.choice(len(lines),1))
            line = lines[rand]
            filename = line[0].split('/')[-1]
            filepath = './data/IMG/'+ filename
            image = cv2.imread(filepath)
            image = cv2.resize(image, (64, 64))
            images.append(image)
            measurements.append(float(line[3]))
        X_valid = np.array(images)
        y_valid = np.array(measurements)
        yield X_valid, y_valid

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, MaxPooling2D, Flatten, Activation, Dense, Cropping2D, Lambda

model = Sequential()
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(64,64,3) ))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile('adam', 'mse')
model.fit_generator(generator_images(lines, 32), samples_per_epoch = 32144, nb_epoch = 2, validation_data=generator_valid(lines), nb_val_samples=6400)

model.save('model.h5')
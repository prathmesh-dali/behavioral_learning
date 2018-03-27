import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
lines = []

X_train = []
y_train = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images_center = []
images_left = []
images_right = []
measurements_center = []
measurements_left = []
measurements_right = []
for line in lines:
    measurement = float(line[3])
    if(measurement < -0.15):
        images_left.append(cv2.imread('./data/IMG/'+ line[0].split('/')[-1]))
        measurements_left.append(measurement)
    elif(measurement > 0.15):
        images_right.append(cv2.imread('./data/IMG/'+ line[0].split('/')[-1]))
        measurements_right.append(measurement)
    else:
        images_center.append(cv2.imread('./data/IMG/'+ line[0].split('/')[-1]))
        measurements_center.append(measurement)

len_left, len_right, len_center = len(images_left), len(images_right), len(images_center)

while len_center - len_left > 0:
    rand = int(np.random.choice(len(lines),1))
    line = lines[rand]
    measurement = float(line[3])
    if(measurement < -0.15):
        images_left.append(cv2.imread('./data/IMG/'+ line[2].split('/')[-1]))
        measurements_left.append(measurement)
    len_left = len(images_left)

while len_center - len_right > 0:
    rand = int(np.random.choice(len(lines),1))
    line = lines[rand]
    measurement = float(line[3])
    if(measurement > 0.15):
        images_right.append(cv2.imread('./data/IMG/'+ line[1].split('/')[-1]))
        measurements_right.append(measurement)
    len_right = len(images_right)

X_train = np.array(images_center+images_left+images_right)
y_train = np.array(measurements_center+measurements_left+measurements_right)

def crop_and_resize(image):
    image = cv2.resize(image[60:140,:], (64,64))
    return image

def randomize_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    multi = random.uniform(0.3,1.0)
    hsv[:,:,2] = multi*hsv[:,:,2]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) 

def generator_images(batchSize = 32):
    while True:
        X_batch = []
        y_batch = []
        images, angle = shuffle(X_train, y_train)
        for i in range(batchSize):
            choice = int(np.random.choice(len(images),1))
            X_batch[i] = randomize_brightness(crop_and_resize(images[choice]))
            y_batch[i] = angle[choice]
            if(random.randint(0,1)):
                X_batch[i] = np.fliplr(X_batch[i])
                y_batch[i] = - y_batch[i]
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        yield X_batch, y_batch

def generator_valid(batchSize = 32):
    while True:
        X_valid = []
        y_valid = []
        for i in range(batchSize):
            choice = int(np.random.choice(len(images),1))
            X_valid[i] = randomize_brightness(crop_and_resize(images[choice]))
            y_valid[i] = angle[choice]
        X_valid = np.array(X_batch)
        y_valid = np.array(y_batch)
        yield X_valid, y_valid

for i in range(10):
    gen = generator_images(32)
    x_b, y_b = next(gen)
    print(x_b.shape, y_b.shape)

# from keras.models import Sequential
# from keras.layers import Convolution2D, Dropout, MaxPooling2D, Flatten, Activation, Dense, Cropping2D, Lambda

# model = Sequential()
# model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(64,64,3) ))
# model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation = 'relu'))
# # model.add(Dropout(0.2))
# # model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation = 'relu'))
# # model.add(Dropout(0.2))
# # model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation = 'relu'))
# # model.add(Dropout(0.2))
# # model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Flatten())
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))

# model.compile('adam', 'mse')
# model.fit_generator(generator_images(lines, 32), samples_per_epoch = 32144, nb_epoch = 2, validation_data=generator_valid(lines), nb_val_samples=6400)

# model.save('model.h5')
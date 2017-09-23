"""based on https://github.com/bmbigbang/nd/blob/master/project-3.py"""
import string
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import Dropout
from keras.models import Sequential, optimizers
import random


def load_images():
    image_names = []
    image_labels = []

    with open('classifier_images/classifier_images.csv', 'r') as csv:
        images_data = csv.readlines()
        # compile the labels from the csv file into a dict with the file name as keys
        for image_data in images_data:
            data = image_data.split(",")
            image_name = data[0]
            image_label = 1.0 if data[1].strip() == 'Green' else 0.0

            image_names.append(image_name)
            image_labels.append(image_label)

    return image_names, image_labels


def process_images(features_in, labels_in, percent_valid=20):
    combined = list(zip(features_in, labels_in))
    random.shuffle(combined)
    features_in[:], labels_in[:] = zip(*combined)
    x_valid_i, y_valid_i = features_in[:int(len(features_in) / percent_valid)], labels_in[
                                                                                :int(len(features_in) / percent_valid)]
    features_i, labels_i = features_in[int(len(features_in) / percent_valid):], labels_in[
                                                                                int(len(features_in) / percent_valid):]

    return features_i, labels_i, x_valid_i, y_valid_i


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def load_image(image_name):
    original_image = cv2.imread('classifier_images/{}'.format(image_name))
    resize_image = cv2.resize(original_image, (160, 80))
    hls_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2HLS)
    final_image = (hls_image.astype(np.float32) / 255.0) + 0.01
    return original_image, final_image


def Generator(X, y, batch_size):
    # depending on the cudnn version and opencv2, sometimes cv2 would crash for me
    # therefore i am importing locally so that it does not conflict with global scope variables
    step = 0
    while True:
        feat, lab = [], np.array([])
        # calculate the step based on batch size and reset to zero if the end of the files is reached
        step = step * batch_size
        if len(X[step:step + batch_size]) < batch_size:
            step = 0
        else:
            step += 1
        for image_name, image_label in zip(X[step:step + batch_size], y[step:step + batch_size]):
            original, final = load_image(image_name)
            feat.append(final)
            lab = np.append(lab, image_label)

        # normalize by diving by (maximum - minimum) after subtracting minimum
        # add a small constant (0.01) to shift away from 0 for better performance operations
        yield np.array(feat), lab


def get_model():
    # implement the nvidia self driving car neural network
    model = Sequential()

    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), batch_input_shape=(None, 80, 160, 3),
                     kernel_initializer='normal', padding='valid', activation='linear'))

    model.add(Conv2D(filters=34, kernel_size=(5, 5), init='normal', strides=(2, 2),
                     padding='valid', activation='linear'))

    model.add(Conv2D(filters=44, kernel_size=(5, 5), init='normal', strides=(2, 2),
                     padding='valid', activation='linear'))

    model.add(Conv2D(filters=52, kernel_size=(3, 3), init='normal', strides=(1, 1),
                     padding='valid', activation='linear'))

    model.add(Conv2D(filters=52, kernel_size=(3, 3), init='normal', strides=(1, 1),
                     padding='valid', activation='linear'))
    model.add(Flatten())
    model.add(Dense(1124, activation='relu', name='dense1'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu', name='dense2'))
    model.add(Dense(50, activation='relu', name='dense3'))
    model.add(Dense(10, activation='relu', name='dense4'))
    model.add(Dense(1, activation='sigmoid', name='dense5'))
    model.summary()
    # use Nadam() with default learning rate parameters for randomized learning rate of small image set
    # allows faster exploration of the hyper parameters by treating learning rate as a parameter to be learnt
    optimizer = optimizers.Adam(lr=0.000005)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def visualize_history(history):
    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def RUN(print_output=True, save_model=True):
    features_temp, labels_temp = load_images()
    features, labels, X_valid, y_valid = process_images(features_temp, labels_temp, 20)
    model = get_model()
    # set image and batch processing parameters
    nb_epoch = 100
    batch_size = 100

    history = model.fit_generator(Generator(features, labels, batch_size), samples_per_epoch=len(features) / batch_size,
                                  nb_epoch=nb_epoch, validation_data=Generator(X_valid, y_valid, batch_size),
                                  validation_steps=len(X_valid) / batch_size)

    if save_model:
        with open('model.json', 'w+') as csv:
            csv.write(model.to_json())
        model.save_weights('model.h5')

    if print_output:
        for image_index in range(len(X_valid)):
            count = 1
            original, final = load_image(X_valid[image_index])
            final_4d = np.expand_dims(final, axis=0)
            network_label = model.predict_classes(final_4d)[0][0]
            actual_label = y_valid[image_index]
            if network_label != actual_label:
                random_name = id_generator()
                file_name = "output/" + str(count) + "_" + str(network_label) + "_" + random_name + ".jpg"
                cv2.imwrite(file_name, original)
                count += 1

    visualize_history(history)

RUN()

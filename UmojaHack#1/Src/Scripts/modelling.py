import cv2
import os
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from keras.applications import InceptionV3
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import (Dense, GlobalAveragePooling1D)
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

top_model_model_path = '../../Output/Models/model.h5'
top_model_weights_path = '../../Output/Models/weights.h5'
train_data_path = '../../Data/Processed/Train'
validation_data_path = '../../Data/Processed/Val'
img_width, img_height = 150, 150
epochs = 10
batch_size = 8


def load_data(dir_path, split=True):
    """

    :type dir_path: str
    :param dir_path: directory path
    :type split: bool
    """
    # initialize the data and labels
    data = []
    labels = []
    dir_labels = ()
    num_class = 0

    # finding the labels
    print("[INFO] Finding Labels...")
    for file in tqdm(os.listdir(dir_path)):
        temp_tuple = (file, 'null')
        dir_labels = dir_labels + temp_tuple
        dir_labels = dir_labels[:-1]
        num_class = num_class + 1

    imagepaths = sorted(list(paths.list_images(dir_path)))

    # loop over the input images
    print("[INFO] Converting images to array...")
    for imagePath in tqdm(imagepaths):
        image = cv2.imread(imagePath)
        # image = cv2.resize(image, (img_height, img_width))
        image = img_to_array(image)
        data.append(image)

        label = imagePath.split(os.path.sep)[-2]

        for i in range(num_class):
            if label == dir_labels[i]:
                label = i
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    if split:
        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.33, random_state=42)

        # convert the labels from integers to vectors
        trainY = to_categorical(trainY, num_classes=num_class)
        testY = to_categorical(testY, num_classes=num_class)

        return trainX, trainY, testX, testY
    else:
        return data, labels


def train(xtrain: list, ytrain: list, xtest: list, ytest: list, xval: list, yval: list):

    base_model = InceptionV3(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    global_average_layer = GlobalAveragePooling1D()
    prediction_layer = Dense(137)
    model = Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    filepath = "model-{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(filepath, period=10)

    # train the network
    print("[INFO] training network...")

    history = model.fit(xtrain, ytrain,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        callbacks=[tensorboard, checkpoint],
                        validation_data=(xtest, ytest))

    model.save_weights(top_model_weights_path)
    model.save(top_model_model_path)

    (eval_loss, eval_accuracy) = model.evaluate(xval, yval, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))
    return history


def make_plot(history):
    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
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
    plt.savefig("../../Output/Figures/perfomance.jpeg")


if __name__ == "__main__":
    xtrain, ytrain, xtest, ytest = load_data(train_data_path, split=True)
    xval, yval = load_data(validation_data_path, split=False)
    history = train(xtrain, ytrain, xtest, ytest, xval, yval)
    make_plot(history)

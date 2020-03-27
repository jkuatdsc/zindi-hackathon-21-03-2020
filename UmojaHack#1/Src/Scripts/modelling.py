import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import (Activation, AveragePooling2D, Flatten, InputLayer,
                          BatchNormalization, Convolution2D, Dense, MaxPooling2D,
                          Dropout, GlobalAveragePooling2D)
from keras.utils.np_utils import to_categorical
from keras.applications import InceptionV3
from keras.callbacks import TensorBoard, ModelCheckpoint
import time
import matplotlib.pyplot as plt

top_model_model_path = '../../Output/Models/model.h5'
top_model_weights_path = '../../Output/Models/weights.h5'
train_data_path = '../../Data/Processed/Train'
validation_data_path = '../../Data/Processed/Val'
img_width, img_height = 512, 384
epochs = 100
batch_size = 8


def train():
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        train_data_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    # nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load('../../Output/Models/bottleneck_features_train.npy')

    # get the class labels for the training data, in the original order
    train_labels = generator.classes

    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator = datagen.flow_from_directory(
        validation_data_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    validation_data = np.load('../../Output/Models/bottleneck_features_validation.npy')

    validation_labels = generator.classes
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)

    # model = applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    # new_input = model.input
    # hidden_layer = model.layers[-1].output
    # model = Model(new)
    # image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    pool_size = (2, 2)

    hidden_num_units = 50
    output_num_units = 10
    input_num_units = 784
    hidden1_num_units = 500
    hidden2_num_units = 500
    hidden3_num_units = 500
    hidden4_num_units = 500
    hidden5_num_units = 500
    output_num_units = 10
    print(train_data.shape)
    model = Sequential([
        Dense(output_dim=hidden1_num_units, input_dim=train_data.shape[1:], activation='relu'),
        Dropout(0.2),
        Dense(output_dim=hidden2_num_units, input_dim=hidden1_num_units, activation='relu'),
        Dropout(0.2),
        Dense(output_dim=hidden3_num_units, input_dim=hidden2_num_units, activation='relu'),
        Dropout(0.2),
        Dense(output_dim=hidden4_num_units, input_dim=hidden3_num_units, activation='relu'),
        Dropout(0.2),
        Dense(output_dim=hidden5_num_units, input_dim=hidden4_num_units, activation='relu'),
        Dropout(0.2),

        Dense(output_dim=output_num_units, input_dim=hidden5_num_units, activation='softmax'),
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3), padding='same'))
    # model.add(GlobalAveragePooling2D())
    # model.add(Activation('softmax', name='predictions'))

    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    filepath = "model-{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(filepath, period=10)

    # train the network
    print("[INFO] training network...")

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        callbacks=[tensorboard, checkpoint],
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)
    model.save(top_model_model_path)

    (eval_loss, eval_accuracy) = model.evaluate(validation_data, validation_labels, batch_size=batch_size, verbose=1)

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
    history = train()
    make_plot(history)

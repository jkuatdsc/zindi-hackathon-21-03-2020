import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
import math

img_width, img_height = 512, 384

train_data_path = '../../Data/Processed/Train'
validation_data_path = '../../Data/Processed/Val'

epochs = 50
batch_size = 8


def save_features() -> bool:
    model = applications.VGG16(include_top=False, weights='imagenet')

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(train_data_path,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode=None,
                                            shuffle=False)

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)
    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(generator, predict_size_train)

    np.save('../../Output/Models/bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(validation_data_path,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode=None,
                                            shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(generator, predict_size_validation)

    np.save('../../Output/Models/bottleneck_features_validation.npy', bottleneck_features_validation)

    return True


if __name__ == "__main__":
    save_features()
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.models import load_model
from keras import applications

img_width, img_height = 512, 384
batch_size = 8

top_model_model_path = '../../Output/Models/model.h5'
top_model_weights_path = '../../Output/Models/weights.h5'
train_data_dir = '../../Data/Processed/Train'

model1 = applications.VGG16(include_top=False, weights='imagenet')
model = load_model(top_model_model_path)
model.load_weights(top_model_weights_path)

datagen = ImageDataGenerator(rescale=1. / 255)
generator = datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical', shuffle=False)
class_dictionary = generator.class_indices


def predict(image_path):
    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model1.predict(image)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)
    probabilities = model.predict_proba(bottleneck_prediction)
    inID = class_predicted[0]
    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]
    return probabilities[0], [k for k, v in class_dictionary.items()]

if __name__ == "__main__":
    print(predict('/home/r0x6f736f646f/Documents/Projects/zindi-hackathon-21-03-2020/UmojaHack#1/Data/Processed/Train/Comanthus_wahlbergii/3IZLBK7.jpeg'))

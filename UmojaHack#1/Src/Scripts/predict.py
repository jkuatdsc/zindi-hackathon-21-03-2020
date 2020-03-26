import numpy as np
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.models import load_model
from keras import applications
import cv2

img_width, img_height = 512, 384
batch_size = 8

top_model_model_path = '../../Output/Models/model.h5'
top_model_weights_path = '../../Output/Models/weights.h5'
train_data_dir = '../../Data/Processed/Train'

model = load_model(top_model_model_path)
model.load_weights(top_model_weights_path)


def predict():
    # load the class_indices saved in the earlier step
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)
    class_dictionary = generator_top.class_indices

    num_classes = len(class_dictionary)

    # add the path to your test image below
    image_path = '../../Data/Processed/Train/Actinoptilum_molle/YVE3N5Q.jpeg'

    orig = cv2.imread(image_path)

    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)

    # build the VGG16 network
    model1 = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model1.predict(image)

    # build top model
    model = load_model(top_model_model_path)
    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)

    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]

    # get the prediction label
    print("Image ID: {}, Label: {}".format(inID, label))

    # display the predictions with the image
    cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

    cv2.imshow("Classification", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    predict()

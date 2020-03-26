import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

img_width, img_height = 150, 150
model_path = 'model.h5'
model_weights_path = 'weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)


def predict(file):
    x = load_img(file, target_size=(img_width, img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    return answer


if __name__ == "__main__":
    print(predict('../../Data/Processed/Train/Actinoptilum_molle/YVE3N5Q.jpeg'))

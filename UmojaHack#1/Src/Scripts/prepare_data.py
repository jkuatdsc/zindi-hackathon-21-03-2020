import numpy as np
import os

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tqdm import tqdm

image_dir = "../../Data/Raw/UmojaHack#1:SAEON_Identifying_marine_invertebrates"
train_dir = image_dir + "/train_small"
test_dir = image_dir + "/test_small"


def augment_images(old_path, new_path):
    image = np.expand_dims(img_to_array(load_img(path=old_path)), axis=0)
    aug = ImageDataGenerator(rescale=1. / 255,
                             rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest'
                             )
    total = 0
    image_gen = aug.flow(image, batch_size=1, save_prefix="gen", save_to_dir=new_path, save_format="jpeg")
    for _ in image_gen:
        total += 1
        if total == 100:
            break


def make_directories():
    print("Making directories")
    if not os.path.exists('../../Data/Processed/Train'):
        os.makedirs('../../Data/Processed/Train')
    if not os.path.exists('../../Data/Processed/Val'):
        os.makedirs('../../Data/Processed/Val')
    if not os.path.exists('../../Data/Processed/Test'):
        os.makedirs('../../Data/Processed/Test')
    for root, dirs, files in os.walk(train_dir):
        for individual_dir in dirs:
            individual_dir = str(individual_dir).replace("(only)", "")
            train_path = '../../Data/Processed/Train/' + str(individual_dir)
            val_path = '../../Data/Processed/Val/' + str(individual_dir)
            os.makedirs(train_path)
            os.makedirs(val_path)


def modify_directories():
    print("Modifying existing directories")
    for root, dirs, files in os.walk(train_dir):
        for individual_dir in dirs:
            path = os.path.join(root, individual_dir)
            m = individual_dir.find("(")
            if m:
                os.rename(path, path.replace(")", "").replace("(", ""))
            if individual_dir == "Nassarius speciosus":
                os.rename(path, path.replace(" ", "_"))


def copy_data():
    print("Copying images to processed directory")
    for root, dirs, files in tqdm(os.walk(train_dir)):
        total_files_number = len(files)
        i = 0
        for file in tqdm(files):
            if file.endswith('jpeg') or file.endswith('JPEG'):
                if (i % 3) == 0:
                    new_path = '../../Data/Processed/Val/' + str(root.split("/")[-1])
                    command = "cp {} {}".format(os.path.join(root, file), new_path)
                    os.system(command)

                else:
                    new_path = '../../Data/Processed/Train/' + str(root.split("/")[-1])
                    command = "cp {} {}".format(os.path.join(root, file), new_path)
                    os.system(command)
                    augment_images(os.path.join(root, file), new_path)
                i += 1

    for root, dirs, files in os.walk(test_dir):
        for file in files:
            new_path = "../../Data/Processed/Test/"
            command = "cp {} {}".format(os.path.join(root, file), new_path)
            os.system(command)


if __name__ == "__main__":
    modify_directories()
    make_directories()
    copy_data()

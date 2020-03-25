import os

image_dir = "../../Data/Raw/UmojaHack#1:SAEON_Identifying_marine_invertebrates"
train_dir = image_dir + "/train_small"
test_dir = image_dir + "/test_small"


def make_directories():
    print("Making directories")
    if not os.path.exists('../../Data/Processed/Train')
        os.makedirs('../../Data/Processed/Train')
    if not os.path.exists('../../Data/Processed/Val')
        os.makedirs('../../Data/Processed/Val')
    if not os.path.exists('../../Data/Processed/Test')
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
    for root, dirs, files in os.walk(train_dir):
        total_files_number = len(files)
        i = 0
        for file in files:
            if file.endswith('jpeg') or file.endswith('JPEG'):
                if (i % 3) == 0:
                    new_path = '../../Data/Processed/Val/' + str(root.split("/")[-1])
                    command = "cp {} {}".format(os.path.join(root, file), new_path)
                    os.system(command)
                else:
                    new_path = '../../Data/Processed/Train/' + str(root.split("/")[-1])
                    command = "cp {} {}".format(os.path.join(root, file), new_path)
                    os.system(command)
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

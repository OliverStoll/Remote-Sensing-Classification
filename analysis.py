import os
import random
import matplotlib.pyplot as plt
import path
import cv2
from PIL import Image


def move_validation_samples(train_path ="data\\Challenge_dataset\\train", parent_path ="data\\Challenge_dataset"):
    """
    Moves the validation samples to the validation folder. This is done one time only.

    :param train_path: path to the training folder
    :param parent_path: path to the parent folder
    """
    # iterate over path and get the num of files for each subfolder
    for subdir, dirs, files in os.walk(train_path):
        print(subdir)
        print(len(files))  # it appears there is a relatively strong class imbalance of the dataset (85-27)

        # for each subfolder, split the data into train and validation 80-20
        num_validation_files = int(len(files) * 0.2)
        print("Thereof validation:", num_validation_files)

        folder_name = subdir.split("\\")[-1]
        # get list of num_validation_files random indexes out of the files list
        validation_indexes = random.sample(range(len(files)), num_validation_files)
        print(validation_indexes)

        # create new folders for validation with random samples from training (val folder was manually created)
        for index in validation_indexes:
            # create validation folder if it doesn't exist
            if not os.path.exists(os.path.join(parent_path, "validation", folder_name)):
                os.makedirs(os.path.join(parent_path, "validation", folder_name))

            file_name = files[index]
            print(file_name)

            # move file to validation folder
            os.rename(os.path.join(subdir, file_name), os.path.join(parent_path, "validation", folder_name, file_name))


# it appears there is a relatively strong class imbalance of the dataset (85-27)
def plot_distribution(train_path="data\\Challenge_dataset\\train"):
    """
    Plots the distribution of the number of files in each subfolder. This shows the balance of the dataset.

    :param train_path: path to the training folder (used before the split of validation)
    """
    list_of_classes = []
    list_of_files = []
    for subdir, dirs, files in os.walk(train_path):
        print(subdir)
        print(len(files))
        list_of_classes.append(subdir.split("\\")[-1])
        list_of_files.append(len(files))
    # plot bargraph with classes and number of files, so that all labels are visible
    plt.figure(figsize=(12, 14))
    plt.bar(list_of_classes, list_of_files)
    plt.setp(plt.gca().get_xticklabels(), fontsize=10, rotation='vertical')
    plt.title("Distribution of classes in training set")
    plt.show()


def print_channels(img_path="data\\Challenge_dataset\\train\\agricultural\\agricultural01.tif"):
    """
    Prints the number of channels in an image. This is useful for checking if the .tif images contain more
    than 3 channels.

    :param img_path: path to the (exemplary) image
    """
    img_pil = Image.open(img_path)
    print('Pillow: ', img_pil.mode, img_pil.size)

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    print('OpenCV: ', img.shape)


if __name__ == "__main__":
    # move_validation_samples()
    # plot_distribution()
    print_channels()

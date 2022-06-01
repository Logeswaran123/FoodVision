"""
FoodVision101

Split Dataset into train and test split
"""
import argparse
import json
import os
import random
import shutil
import tqdm

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c",  "--classes",  required=True, help="Path to classes text file", type=str)
ap.add_argument("-img", "--images", required=True, help="Path to raw images", type=str)
args = vars(ap.parse_args())

# Get labels
def get_labels(label_path):
    """
    Accepts a label path (in the form of a JSON) and returns the file
    as a Python object.
    """
    with open(label_path) as file:
        return json.load(file)

def copy_images(parent_folder, new_subset, dataset, target_labels):
    """
    Copies `labels[target_labels]` images from `parent_folder` to
    `new_subset` (named after `dataset`) folder.

    E.g. move steak images to data/steak_subset/train/ &
    data/steak_subset/test/

    Args:
    --------
    parent_folder (str)  - original folder path with all data
    new_subset (str)     - name of parent folder to copy to
    dataset (str)        - which dataset? (train or test)
    labels (list)        - list of training or test labels
    target_labels (list) - list of target labels to copy e.g. ["steak", "pizza"]
    """
    # Get the appropriate labels
    print(f"\nUsing {dataset} labels...")
    meta_folder = parent_folder + "\\meta\\meta\\"
    labels = get_labels(meta_folder + dataset + ".json")

    # Loop through target labels
    for i in target_labels:
        # Make target directory
        os.makedirs(parent_folder + "\\" + new_subset + "\\" + dataset + "\\" + i,
                    exist_ok=True)

        # Go through labels and get appropriate classes
        images_moved = [] # Keep track of images moved
        for j in labels[i]:
            # Create original image path and new path
            og_path = parent_folder + "\\images\\" + j + ".jpg"
            new_path = parent_folder + "\\" + new_subset + "\\" + dataset + "\\" + j + ".jpg"

            # Copy images from old path to new path
            shutil.copy2(og_path, new_path)
            images_moved.append(new_path)
        print(f"Copied {len(images_moved)} images from {dataset} dataset {i} class...")

def get_percent_images(target_dir, new_dir, sample_amount=0.1, random_state=42):
    """
    Get sample_amount percentage of random images from target_dir and copy them to new_dir.

    Preserves subdirectory file names.

    E.g. target_dir=pizza_steak/train/steak/all_files
                -> new_dir_name/train/steak/X_percent_of_all_files

    Parameters
    --------
    target_dir (str) - file path of directory you want to extract images from
    new_dir (str) - new directory path you want to copy original images to
    sample_amount (float), default 0.1 - percentage of images to copy (e.g. 0.1 = 10%)
    random_state (int), default 42 - random seed value
    """
    # Set random seed for reproducibility
    random.seed(random_state)

    # Get a list of dictionaries of image files in target_dir
    # e.g. [{"class_name":["2348348.jpg", "2829119.jpg"]}]
    images = [{dir_name: os.listdir(target_dir + dir_name)} for dir_name in os.listdir(target_dir)]

    for i in images:
        for k, v in i.items():
            # How many images to sample?
            sample_number = round(int(len(v)*sample_amount))
            print(f"There are {len(v)} total images in '{target_dir+k}' so we're going to copy {sample_number} to the new directory.")
            print(f"Getting {sample_number} random images for {k}...")
            random_images = random.sample(v, sample_number)

            # Make new dir for each key
            new_target_dir = new_dir + k
            print(f"Making dir: {new_target_dir}")
            os.makedirs(new_target_dir, exist_ok=True)

            # Keep track of images moved
            images_moved = []

            # Create file paths for original images and new file target
            print(f"Copying images from: {target_dir}\n\t\t to: {new_target_dir}/\n")
            for file_name in tqdm(random_images):
                og_path = target_dir + k + "/" + file_name
                new_path = new_target_dir + "/" + file_name

                # Copy images from OG path to new path
                shutil.copy2(og_path, new_path)
                images_moved.append(new_path)

            # Make sure number of images moved is correct
            assert len(os.listdir(new_target_dir)) == sample_number
            assert len(images_moved) == sample_number


def main():
    # Get all classnames
    classes = []
    with open(args["classes"]) as file:
        for line in file.readlines():
            classes.append(line.split("\n")[0])

    assert len(classes) == 101

    parent_folder = args["images"]
    new_subset = "train_test_split"
    datasets = ["train", "test"]

    # Copy training/test images
    for i in datasets:
        copy_images(parent_folder=parent_folder,
                    new_subset=new_subset,
                    dataset=i,
                    target_labels=classes)


if __name__ == "__main__":
    main()

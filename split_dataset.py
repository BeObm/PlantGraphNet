import os
import random
import shutil

def set_seed():
    random.seed(42)

def split_image_dataset(data_path, train_ratio=0.8, val_ratio=0, test_ratio=0.2):
    if not os.path.exists(data_path):
        raise ValueError(f"The specified data path {data_path} does not exist.")

    set_seed()
    # path to destination folders
    train_folder = os.path.join(os.path.dirname(data_path), 'train')
    val_folder = os.path.join(os.path.dirname(data_path), 'val')
    test_folder = os.path.join(os.path.dirname(data_path) , 'test')

    # Define a list of image extensions
    image_extensions = ['.jpg', '.jpeg', '.png']

    # Create destination folders if they don't exist
    for folder_path in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Loop through each class folder in the dataset path
    for class_name in os.listdir(data_path):
        class_folder = os.path.join(data_path, class_name)
        
        # Skip if it's not a directory (we are only interested in class directories)
        if not os.path.isdir(class_folder):
            continue

        # Create corresponding folders in train, val, and test directories
        class_train_folder = os.path.join(train_folder, class_name)
        class_val_folder = os.path.join(val_folder, class_name)
        class_test_folder = os.path.join(test_folder, class_name)

        for folder in [class_train_folder, class_val_folder, class_test_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # List all images in the class folder
        imgs_list = [filename for filename in os.listdir(class_folder) if os.path.splitext(filename)[-1] in image_extensions]

        # Shuffle the list of image filenames
        random.shuffle(imgs_list)

        # Determine the number of images for each set
        train_size = int(len(imgs_list) * train_ratio)
        val_size = int(len(imgs_list) * val_ratio)
        test_size = int(len(imgs_list) * test_ratio)

        # Copy images to the appropriate train, validation, and test folders
        for i, f in enumerate(imgs_list):
            if i < train_size:
                dest_folder = class_train_folder
            elif i < train_size + val_size:
                dest_folder = class_val_folder
            else:
                dest_folder = class_test_folder

            try:
                shutil.copy(os.path.join(class_folder, f), os.path.join(dest_folder, f))
                print(f"Successfully copied {f} to {dest_folder}")
            except Exception as e:
                print(f"Error copying {f}: {e}")

    return train_folder, val_folder, test_folder



if __name__ == "__main__":
    data_path = "dataset/images/lidl"
    # # data_path = "dataset\images\lild_dl_test"
    train_folder,val_folder,test_folder = split_image_dataset(data_path,train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    print(f"Data split successfully into train, validation, and test folders: {train_folder}, {val_folder}, {test_folder}")
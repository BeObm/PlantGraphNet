import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.filters import sobel
from torch.utils.data import Dataset, DataLoader


def load_image(image_path):
    """ Load image and convert to tensor """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb


def extract_color_histogram(img, bins=(8, 8, 8)):
    """ Extract color histogram features """
    hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_texture_features(gray_img):
    """ Extract texture features using GLCM """
    glcm = graycomatrix(gray_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation])


def extract_edge_features(gray_img):
    """ Extract edge features using Canny Edge Detector """
    edges = cv2.Canny(gray_img, 100, 200)
    edge_count = np.sum(edges) / 255
    return np.array([edge_count])


def extract_hog_features(gray_img):
    """ Extract HOG (Histogram of Oriented Gradients) features """
    features, _ = hog(gray_img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=False)
    return features


def extract_lbp_features(gray_img, P=8, R=1):
    """ Extract Local Binary Pattern (LBP) features """
    lbp = local_binary_pattern(gray_img, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype(float) / hist.sum()  # Normalize histogram
    return hist


def extract_sobel_features(gray_img):
    """ Extract Sobel edge features """
    sobel_img = sobel(gray_img)
    return np.array([np.mean(sobel_img), np.std(sobel_img)])


class ImageFeatureDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        img, img_rgb = load_image(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract features
        color_hist = extract_color_histogram(img)
        texture_features = extract_texture_features(gray_img)
        edge_features = extract_edge_features(gray_img)
        hog_features = extract_hog_features(gray_img)
        lbp_features = extract_lbp_features(gray_img)
        sobel_features = extract_sobel_features(gray_img)

        # Convert image to tensor
        if self.transform:
            img_tensor = self.transform(img_rgb)
        else:
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor()])
            img_tensor = transform(img_rgb)

        # Concatenate all extracted features
        additional_features = np.hstack((color_hist, texture_features, edge_features, hog_features, lbp_features, sobel_features))
        additional_features_tensor = torch.tensor(additional_features, dtype=torch.float)

        return img_tensor, additional_features_tensor, torch.tensor(label, dtype=torch.long)


# Example usage
def create_dataloader(image_folder, labels_dict, batch_size=16):
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]
    labels = [labels_dict[os.path.basename(img)] for img in image_paths]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImageFeatureDataset(image_paths, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Example data directory and labels
data_dir = "path_to_images"
labels_dict = {"image1.jpg": 0, "image2.jpg": 1}  # Example labels

dataloader = create_dataloader(data_dir, labels_dict)

# Iterate through the dataset
for images, features, labels in dataloader:
    print("Image Tensor Shape:", images.shape)
    print("Additional Features Shape:", features.shape)
    print("Labels:", labels)





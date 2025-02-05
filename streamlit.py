import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from main_cnn_model import CNNModel

# Constants - List of Pretrained Models
PRETRAINED_MODELS = {
    "GoogleNet": "GoogleNet_weights.pth",
    "Custom CNN Model": "Our_CNN_Model_weights.pth",
    "ResNet50": "ResNet50_weights.pth",
}

# Streamlit app title
st.title("Image Classification with Multiple Models")


# Function to load model dynamically based on user selection
@st.cache_resource
def load_model(model_name):
    if model_name == "Faster R-CNN":
        model = CNNModel()  # Replace with Faster R-CNN instantiation as needed.
    elif model_name == "ResNet50":
        from torchvision.models import resnet50
        model = resnet50(pretrained=False)
    elif model_name == "Custom CNN Model":
        model = CNNModel()  # Your custom CNN model.
    else:
        st.error(f"Model {model_name} not supported!")
        return None

    # Load weights
    model_weights_path = PRETRAINED_MODELS[model_name]
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    return model


# Function for image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).unsqueeze(0)


# Function to make predictions
def perform_inference(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs


# Draw function to display results
def draw_boxes_on_image(image, boxes, labels, scores):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        (x_min, y_min, x_max, y_max) = box
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x_min, y_min - 10, f"{label}: {score:.2f}",
            color='red', fontsize=12, backgroundcolor='white'
        )

    st.pyplot(fig)


# Function to map labels to class names
@st.cache_data
def load_class_names():
    return [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
        'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]


# User Interface
model_name = st.selectbox("Choose a Model", list(PRETRAINED_MODELS.keys()))
uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])

# Main logic
if uploaded_file and model_name:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    preprocessed_image = preprocess_image(image)

    # Load selected model and make predictions
    model = load_model(model_name)
    if model:
        outputs = perform_inference(model, preprocessed_image)

        # Extract predictions
        boxes = outputs[0]['boxes'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()

        # Map labels to names
        coco_classes = load_class_names()
        label_names = [coco_classes[label] for label in labels]

        # Draw results
        draw_boxes_on_image(np.array(image), boxes, label_names, scores)

        # Display confidence scores
        st.write("Confidence Scores:")
        for label, score in zip(label_names, scores):
            st.write(f"{label}: {score:.2f}")

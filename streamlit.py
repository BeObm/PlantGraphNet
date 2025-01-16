import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the model (Faster R-CNN in this example)
st.title("Image Detection with PyTorch and Streamlit")

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

model = load_model()

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

def draw_boxes(image, boxes, labels, scores):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        if score >= 0.5:  # Confidence threshold
            (x_min, y_min, x_max, y_max) = box
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x_min, y_min - 10, f"{label}: {score:.2f}",
                color='red', fontsize=12, backgroundcolor='white'
            )
    st.pyplot(fig)

uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'png', 'jpeg'])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        outputs = model(img_tensor)

    # Extract data
    boxes = outputs[0]['boxes'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    # Load COCO class names
    @st.cache_data
    def load_coco_classes():
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

    coco_classes = load_coco_classes()

    # Map labels to class names
    label_names = [coco_classes[label] for label in labels]

    # Draw boxes and show image
    draw_boxes(np.array(image), boxes, label_names, scores)

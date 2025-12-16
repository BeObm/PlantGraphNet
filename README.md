# PlantGraphNet

**PlantGraphNet** is a hybrid **CNN–GNN framework for fine-grained plant species classification**, specifically designed to capture **both visual appearance and plant morphological structure** from high-resolution imagery. The method was introduced in the paper:

> **Oloulade et al. (2025)** – *Hybrid CNN–GNN Architectures with Distributed Training for Heathland Plant Classification*

PlantGraphNet was developed for **ecological monitoring and biodiversity assessment**, with a focus on **heathland ecosystems**, where plant species exhibit complex, irregular spatial structures that are poorly captured by grid-based CNNs alone.

The core idea is to:

* represent each plant image as a **graph of morphological keypoints** (nodes = SIFT keypoints, edges = spatial relations),
* learn **relational structure** with a Graph Neural Network (GNN),
* extract **global visual context** with a CNN backbone (ResNet-50), and
* **fuse both modalities** into a unified, interpretable representation.

This README aligns the codebase explicitly with the methodology, dataset, and experiments described in the paper.

---

## 1. Project Structure

```
PlantGraphNet/
├── Build_graph_dataset.py    # Graph construction from plant images (SIFT + kNN)
├── split_dataset.py         # Train / validation / test split
├── main_gnn.py              # Main training and evaluation entry point
├── train_test_model.py      # Training and evaluation loops
├── model.py                 # Hybrid CNN–GNN model definitions
├── utils.py                 # Utilities (metrics, logging, data loading)
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```

### Design philosophy

* **Modular**: graph construction, modeling, and training are decoupled
* **Reproducible**: fixed splits, explicit baselines
* **Scalable**: supports distributed data-parallel (DDP) training

---

## 2. Dataset Description (as used in the paper)

### 2.1 Study site and imagery

The dataset consists of **high-resolution RGB aerial imagery** collected over a **12-hectare coastal heathland** at *Vust Heath, Northern Jutland, Denmark* (NATURA 2000 area). Images were acquired using a drone-mounted RGB camera and stitched into a georeferenced orthomosaic with a **ground sampling distance (GSD) of 1.2 cm per pixel**.


### 2.2 Annotation and classes

* Expert ecologists manually annotated the orthomosaic using GIS tools.
* Polygons were labeled into **10 land-cover / vegetation classes**:

  * amm, calluna, empetrum, grass, lichens, myrica,
  * nonveg, rosa rugosa, salix, trees
* Annotated regions were cropped into **150 × 150 pixel image patches**.

### 2.3 Dataset split

The dataset contains **12,140 image patches**, split as:

* 60% training (7,281 images)
* 20% validation (2,424 images)
* 20% testing (2,435 images)

Standard data augmentation (rotation, flipping, zooming) is applied during training.

---

## 3. Graph Dataset Construction

### 3.1 Keypoint-based graph representation

Each image patch is converted into a **graph** following Algorithm 1 in the paper :

* **Nodes**: SIFT keypoints detected in the image
* **Node features**: 128-dimensional SIFT descriptors
* **Edges**: k-nearest-neighbour (k = 5) connections in 2D image space
* **Edge attributes**: constant weight (1.0)
* **Graph label**: plant species class

This construction preserves **local morphology and spatial relationships**, enabling relational learning beyond pixel grids.

### 3.2 Build graph dataset

```bash
python Build_graph_dataset.py \
    --data_dir path/to/image_patches \
    --output_dir path/to/graph_dataset
```

The script:

1. Loads each image
2. Extracts SIFT keypoints and descriptors
3. Constructs a kNN graph
4. Saves a PyTorch Geometric `Data` object

---



## 5. Model Architecture

### 5.1 Graph branch (GNN)

* Three graph convolution layers
* Normalization + dropout + nonlinearity at each layer
* Global additive pooling to obtain graph-level embedding

### 5.2 Image branch (CNN)

* ResNet-50 pretrained on ImageNet-1K
* Global average pooling to extract image-level features
* Linear projection into shared embedding space

### 5.3 Multimodal fusion

* Concatenation of graph and image embeddings
* Dropout regularization (p = 0.2)
* Linear classifier for final prediction

Unimodal variants (CNN-only, GNN-only) are also supported for ablation studies.

---

## 6. Training and Evaluation

### 6.1 Standard training

```bash
python main_gnn.py \
    --dataset_dir path/to/graph_dataset \
    --model plantgraphnet \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001
```

* Optimizer: Adam
* Loss: categorical cross-entropy
* Metrics: accuracy, precision, recall, F1-score

### 6.2 Distributed Data-Parallel (DDP) training

The code supports **multi-GPU training** using PyTorch Distributed Data-Parallel:

```bash
torchrun --nproc_per_node=4 main_gnn.py --model plantgraphnet
```

DDP provides **near-linear speed-up** while preserving convergence, as shown in the paper.

---

## 7. Baseline Models

The following CNN baselines are implemented and evaluated under identical settings :

* AlexNet
* VGG16 / VGG19
* ResNet50 / ResNet101
* MobileNetV2
* GoogleNet
* YOLOv8 (classification head)

### Running a baseline

```bash
python main_gnn.py \
    --dataset_dir path/to/graph_dataset \
    --model resnet50
```

Backbone weights are ImageNet-pretrained and frozen; only classification heads are trained.

---

## 8. Running on a New Dataset

To apply PlantGraphNet to a new plant or ecological dataset:

1. **Prepare image patches** with consistent size
2. **Ensure class labels** are defined
3. Modify `Build_graph_dataset.py` if:

   * a different keypoint detector is needed
   * alternative graph construction (RAG, superpixels) is desired
4. Build graphs, split data, and train as usual

The framework is adaptable to:

* other vegetation types
* multi-label classification
* structurally complex biological imagery

---

## 9. Reproducibility and Best Practices

* Fix random seeds
* Use identical splits for all models
* Log hyperparameters and checkpoints
* Report class-wise metrics and confusion matrices

---

## 10. Limitations and Extensions

Current limitations (as discussed in the paper):

* Fixed kNN graph construction
* Static 2D imagery only

Possible extensions:

* Adaptive or learned graph connectivity
* Temporal or 3D plant modeling
* Multilabel and open-world classification
* Integration with segmentation models (e.g., SAM)

---

## 11. Citation

If you use this code, please cite:

Oloulade, B. M., et al. *Hybrid CNN–GNN Architectures with Distributed Training for Heathland Plant Classification*, 2025.

---

## 12. Contact

For questions or collaboration, please contact the corresponding author.

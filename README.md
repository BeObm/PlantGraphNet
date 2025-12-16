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




## 4. Training and Evaluation

### 4.1 Standard training

```bash
python main_gnn.py 
```


### 4.2 Distributed Data-Parallel (DDP) training

The code supports **multi-GPU training** using PyTorch Distributed Data-Parallel:

```bash
torchrun --nproc_per_node=4 main_gnn.py 
```

DDP provides **near-linear speed-up** while preserving convergence, as shown in the paper.

---

## 5. Baseline Models

The following CNN baselines are implemented and evaluated under identical settings :

* AlexNet
* VGG16 / VGG19
* ResNet50 / ResNet101
* MobileNetV2
* GoogleNet
* YOLOv8 (classification head)

### Running a baseline

```bash
python Baselines/baseline_main.py \
    --model_name AlexNet
```
---

## 6. Running on a New Dataset

To apply PlantGraphNet to a new plant or ecological dataset:

1. **Prepare image patches** with consistent size
2. **Ensure class labels** are defined
3. Modify `Build_graph_dataset.py` if:

   * a different keypoint detector is needed
   * alternative graph construction (RAG, superpixels) is desired
4. Build graphs, split data, and train as usual

---



## 7. Limitations and Extensions

Current limitations (as discussed in the paper):

* Fixed kNN graph construction
* Static 2D imagery only

Possible extensions:

* Adaptive or learned graph connectivity
* Temporal or 3D plant modeling
* Multilabel and open-world classification
* Integration with segmentation models

---

## 8. Citation

If you use this code, please cite:

Oloulade, B. M., et al. *Hybrid CNN–GNN Architectures with Distributed Training for Heathland Plant Classification*, 2025.

---

## 9. Contact

For questions or collaboration, please send an email to b.oloulade@ecos.au.dk.

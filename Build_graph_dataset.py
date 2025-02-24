import argparse
from model import *
import importlib
from datetime import datetime
if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser()

    parser.add_argument("--type_graph", default="grid", help="define how to construct nodes and egdes", choices=["harris", "grid", "multi","grid2"])
    parser.add_argument("--apply_transform", default=True, type=bool, help="apply transform", choices=[True, False])
    parser.add_argument("--images_per_class", type=int, default=0, help="number of images to use for training/test per class; 0 means all")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--connectivity", type=str, default="4-connectivity", help="connectivity", choices=["4-connectivity", "8-connectivity"])
    parser.add_argument("--use_image_feats", default=True, type=bool, help="use input  image features as graph feature or not")

    args = parser.parse_args()


    create_config_file(args.type_graph, args.connectivity)
    graph_constructor_obj = importlib.import_module(f"build_dataset.{args.type_graph}")
    graph_constructor = getattr(graph_constructor_obj, "build_dataset")
    start_time = datetime.now()
    
    print("Creating training graph datasets...")
    graph_constructor(dataset_path="dataset/images/train",
                                     args=args,
                                     type_dataset="train",
                                     apply_transform=True)
    print("Creating validation graph datasets...")
    graph_constructor(dataset_path="dataset/images/val",
                                     args=args,
                                     type_dataset="val",
                                     apply_transform=False)
    print("Creating testing graph datasets...")
    graph_constructor(dataset_path="dataset/images/test",
                                     args=args,
                                     type_dataset="test",
                                     apply_transform=False)

   
    print("Graph datasets created successfully in ", datetime.now() - start_time)
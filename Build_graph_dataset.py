import argparse
from model import *
import importlib
from datetime import datetime
if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help="type dataset", default="train")
    parser.add_argument("--type_graph", default="grid", help="define how to construct nodes and egdes", choices=["harris", "grid", "multi","grid2"])
    parser.add_argument("--apply_transform", default=True, type=bool, help="apply transform", choices=[True, False])
    parser.add_argument("--images_per_class", type=int, default=20, help="number of images to use for training/test per class; 0 means all")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")

    args = parser.parse_args()


    create_config_file(args.dataset, args.type_graph)
    graph_constructor_obj = importlib.import_module(f"build_dataset.{args.type_graph}")
    graph_constructor = getattr(graph_constructor_obj, "build_dataset")
    start_time = datetime.now()
    dataset_path = graph_constructor(dataset_path=config['param']["image_dataset_root"],
                                     nb_per_class=args.images_per_class,
                                     apply_transform=args.apply_transform)
    print("Graph dataset created successfully in ", datetime.now() - start_time)


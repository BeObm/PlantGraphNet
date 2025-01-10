import argparse
from model import *
import importlib

if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help="Dataset name", default="train")
    parser.add_argument("--type_graph", default="harris", help="define how to construct nodes and egdes",
                        choices=["harris", "grid", "multi"])
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")

    args = parser.parse_args()
    create_config_file(args.dataset, args.type_graph)
    graph_constructor_obj = importlib.import_module(f"build_dataset.{args.type_graph}")
    graph_constructor = getattr(graph_constructor_obj, "build_dataset")

    dataset_path = graph_constructor(config['param']["image_dataset_root"],config['param']["graph_dataset_name"])


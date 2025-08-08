import argparse
from datetime import datetime
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, ResGatedGraphConv, GATConv, GATv2Conv, TransformerConv, \
    TAGConv, ARMAConv, SGConv, SSGConv, MFConv, GMMConv, SplineConv, NNConv, FeaStConv, LEConv
from torch_geometric.nn import PNAConv, ClusterGCNConv, PANConv, SuperGATConv, FAConv, EGConv, GeneralConv, MixHopConv
from tqdm import tqdm
import torch.multiprocessing as mp
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim.lr_scheduler as lr_scheduler
from model import *
import os
import gc
from copy import deepcopy
from accelerate import Accelerator, InitProcessGroupKwargs

if __name__ == "__main__":
    set_seed()
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(backend="gloo")])

    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name_file', type=str, default='dataset/class_names.json')
    parser.add_argument("--type_graph", default="superpixel_graph", help="define how to construct nodes and egdes",
                        choices=["grid_graph", "superpixel_graph", "keypoint_graph", "region_adjacency_graph",
                                 "feature_map_graph", "mesh3d_graph", "multi_graphs"])
    parser.add_argument("--use_image_feats", default=True, help="use input  image features as graph feature or not")
    parser.add_argument("--image_backbone", default="yolo8n", help="which backbone to use",
                        choices=["VGG19", "VGG16", "ResNet50", "ResNet101", "AlexNet", "MobileNetV2", "GoogleNet",
                                 "Unet", "yolo8", "yolo11", "yolo12"])
    parser.add_argument("--hidden_dim", default=64, type=int, help="hidden_dim")
    parser.add_argument("--num_epochs", type=int, default=20, help="num_epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="batch_size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning_rate")
    parser.add_argument("--wd", type=float, default=0.0001, help="wd")
    parser.add_argument("--Conv1", default=LEConv, help="Conv1")
    parser.add_argument("--Conv2", default=LEConv, help="Conv2")
    parser.add_argument("--nb_gpus", default=4, type=int, help="number of GPUs")
    parser.add_argument("--connectivity", type=str, default="4-connectivity", help="connectivity",
                        choices=["4-connectivity", "8-connectivity"])

    args = parser.parse_args()
    start_time = datetime.now()
    args.batch_size = int(args.batch_size) * int(args.nb_gpus)
    create_config_file(args)
    args.result_dir = config['param']['result_folder']
    class_names = get_class_names(args.class_name_file)
    print(f" {'*' * 10}  Loading graph datasets...")
    train_graph_list, feat_size = Load_graphdata(f"dataset/graphs/{args.type_graph}/train")

    train_split, temp_split = train_test_split(train_graph_list, test_size=0.2, random_state=42)
    val_split, test_split = train_test_split(temp_split, test_size=0.5, random_state=42)

    # Final loaders
    train_loader = graphdata_loader(train_split, batch_size=args.batch_size, type_data="train")
    val_loader = graphdata_loader(val_split, batch_size=args.batch_size, type_data="val")
    test_loader = graphdata_loader(test_split, batch_size=args.batch_size, type_data="test")

    # Print stats
    print(f"Number of training graphs: {len(train_split)}")
    print(f"Number of validation graphs: {len(val_split)}")
    print(f"Number of testing graphs: {len(test_split)}")

    input_dim = feat_size
    hidden_dim = args.hidden_dim
    output_dim = len(class_names)
    print(f"Number of classes: {len(class_names)}")
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    set_seed()
    if args.type_graph == "multi_graphs":
        pass
    else:
        model = GNNModel(num_node_features=input_dim,
                         hidden_dim=hidden_dim,
                         num_classes=output_dim,
                         Conv1=args.Conv1,
                         Conv2=args.Conv2,
                         image_feature=67500,
                         use_image_feats=args.use_image_feats,
                         image_backbone=args.image_backbone)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters in the model: {pytorch_total_params}")
    add_config("param", "Model parameters", pytorch_total_params)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_accuracies = []
    device = accelerator.device

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.01)

    model, optimizer, scheduler, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer,
                                                                                             scheduler, train_loader,
                                                                                             val_loader, test_loader)

    train_losses, model = train_function(epochs=num_epochs,
                                         model=model,
                                         train_dataloader=train_loader,
                                         val_dataloader=val_loader,
                                         type_graph=args.type_graph,
                                         criterion=criterion,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         accelerator=accelerator)

    print(f"Time taken to train the model: {datetime.now() - start_time}")

    add_config("param", "Training time", datetime.now() - start_time)
    add_config("param", "Best Loss", min(train_losses))

    plot_and_save_training_performance(losses=train_losses, args=args)

    cls_report, hammingloss = test_function(accelerator=accelerator,
                                            model=model,
                                            test_loader=test_loader,
                                            class_names=class_names)

    cr = pd.DataFrame(cls_report).transpose()
    cr["hammingloss"] = hammingloss
    cr.to_excel(f"{config['param']['result_folder']}/result_for_GNN_Model.xlsx")

    print(f"Model Classification report for GNN model \n ")
    print(cr)


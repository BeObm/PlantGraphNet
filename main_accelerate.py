import argparse
from datetime import datetime
from torch_geometric .nn import GCNConv, SAGEConv,GraphConv,ResGatedGraphConv,GATConv,GATv2Conv,TransformerConv,TAGConv,ARMAConv,SGConv,SSGConv,MFConv,GMMConv,SplineConv,NNConv,FeaStConv,LEConv
from torch_geometric.nn import PNAConv,ClusterGCNConv,PANConv,SuperGATConv,FAConv,EGConv,GeneralConv,MixHopConv
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from model import *
import os
import gc
from copy import deepcopy
from accelerate import Accelerator, InitProcessGroupKwargs

   
if __name__ == "__main__":
    set_seed()
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(backend="gloo")])

    parser = argparse.ArgumentParser()

    parser.add_argument("--type_graph", default="superpixel_graph", help="define how to construct nodes and egdes", choices=["grid_graph", "superpixel_graph", "keypoint_graph", "region_adjacency_graph", "feature_map_graph","mesh3d_graph", "multi_graphs"])
    parser.add_argument("--use_image_feats", default=True, type=bool, help="use input  image features as graph feature or not")
    parser.add_argument("--hidden_dim", default=256, type=int, help="hidden_dim")
    parser.add_argument("--num_epochs", type=int, default=200, help="num_epochs")
    parser.add_argument("--batch_size", type=int, default=4*32, help="batch_size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning_rate")
    parser.add_argument("--wd", type=float, default=0.0001, help="wd")
    parser.add_argument("--Conv1", default=LEConv, help="Conv1")
    parser.add_argument("--Conv2", default=LEConv, help="Conv2")
    parser.add_argument("--nb_gpus", default=4, help="number of GPUs")
    parser.add_argument("--connectivity", type=str, default="4-connectivity", help="connectivity", choices=["4-connectivity", "8-connectivity"])
    
    args = parser.parse_args()
    start_time=datetime.now()

    create_config_file(args.type_graph)
   
    print(f" {'*'*10}  Loading training graph datasets...")
    train_graph_list,feat_size,class_names = Load_graphdata(f"dataset/graphs/{args.type_graph}/train")
    print(f" {'*'*10}  Loading testing graph datasets...")
    test_graph_list,_,_ = Load_graphdata(f"dataset/graphs/{args.type_graph}/test")
    print(f" {'*'*10}  Loading validation graph datasets...")
    val_graph_list,_,_ = Load_graphdata(f"dataset/graphs/{args.type_graph}/val")
    
    print(f"Number of training graphs: {len(train_graph_list)}")
    print(f"Number of testing graphs: {len(test_graph_list)}")
    print(f"Number of validation graphs: {len(val_graph_list)}")
    
    train_loader= graphdata_loader(train_graph_list,batch_size=args.batch_size,type_data="train")
    test_loader=graphdata_loader(test_graph_list,batch_size=args.batch_size,type_data="test")

    
    input_dim = feat_size
    hidden_dim = args.hidden_dim
    output_dim = 10
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    set_seed()
    
    if args.type_graph == "multi_graphs":
        model=MultiGraphModel(num_node_features=input_dim,
                              hidden_dim=hidden_dim,
                              num_classes=output_dim,
                              Conv1=args.Conv1,
                              Conv2=args.Conv2,
                              image_feature=67500,
                              use_image_feats=args.use_image_feats)
    else:
        model = GNNModel(num_node_features=input_dim,
                     hidden_dim=hidden_dim,
                     num_classes=output_dim,
                     Conv1=args.Conv1,
                     Conv2=args.Conv2,
                     image_feature=67500,
                     use_image_feats=args.use_image_feats)
    
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters in the model: {pytorch_total_params}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=args.wd)
    pbar = tqdm(num_epochs)
    pbar.set_description("training model")
    best_loss=99999
    train_losses = []
    train_accuracies = []
    device = accelerator.device
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader,test_loader)


    for epoch in range(1,num_epochs+1):
        loss = train_function(model=model,
                              dataloader=train_loader, 
                              optimizer=optimizer, 
                              criterion=criterion,
                              accelerator=accelerator)
        train_losses.append(loss)
        if loss <= best_loss:
            best_loss = loss
            best_model=deepcopy(model)
        if epoch % 10 == 0:
            torch.save(best_model.state_dict(), f"results/GNN_Models/{args.type_graph}_best_model.pth")
        pbar.write(f'\n Epoch [{epoch}/{num_epochs}]: Loss: {round(loss, 5)} | Current best loss: {round(best_loss, 5)}')
        pbar.update(1)
        
    model=best_model
    print(f"Time taken to train the model: { datetime.now() - start_time}")

    plot_and_save_training_performance(num_epochs=num_epochs,
                                       losses=train_losses,
                                       folder_name=config['param']['result_folder'])

    cls_report = test_function(accelerator=accelerator,
                               model=model,
                               test_loader=test_loader,
                               class_names=class_names)

    cr = pd.DataFrame(cls_report).transpose()
    cr.to_excel( f"{config['param']['result_folder']}/result_for_GNN_Model.xlsx")
    
    print(f"Model Classification report for GNN model \n ")
    print(cr)
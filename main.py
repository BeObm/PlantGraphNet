import argparse
from datetime import datetime
from torch_geometric.nn import GENConv, GATConv
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import *
import os



def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'    # shouyld be the ip of the master node
    os.environ['MASTER_PORT'] = '12355'  #can be any free port on my pc
    # initialize the process group
    init_process_group("gloo", rank=rank, world_size=world_size)
    # Explicitly setting seed to make sure that models created in two processes start from same random weights and biases.
    set_seed()
    
    
if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help="Dataset name", default="train")
    parser.add_argument("--type_graph", default="harris", help="define how to construct nodes and egdes", choices=["harris", "grid", "multi"])
    parser.add_argument("--use_image_feats", default=True, type=bool, help="use input  image features as graph feature or not")
    parser.add_argument("--hidden_dim", default=64, type=int, help="hidden_dim")
    parser.add_argument("--num_epochs", type=int, default=100, help="num_epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning_rate")
    parser.add_argument("--wd", type=float, default=0.005, help="wd")
    parser.add_argument("--Conv1", default=GENConv, help="Conv1")
    parser.add_argument("--Conv2", default=GATConv, help="Conv2")
    parser.add_argument("--gpu_idx", default=3, help="GPU  num")
    parser.add_argument("--connectivity", type=str, default="4-connectivity", help="connectivity", choices=["4-connectivity", "8-connectivity"])
    
    args = parser.parse_args()
    ddp_setup(rank=args.gpu_idx, world_size=4)

    create_config_file(args.dataset, args.type_graph, args.connectivity)
    # device = torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')
    print("Rank:",args.gpu_idxice)
    start_time=datetime.now()
    train_loader,feat_size,class_names= Load_graphdata(config['param']["graph_dataset_folder"],args=args)
    test_loader,_,_ =Load_graphdata(config['param']["graph_dataset_folder"],args=args)

    print(f"The labels are {class_names}")

    input_dim = feat_size
    hidden_dim = args.hidden_dim
    output_dim = 10
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    model = GNNModel(num_node_features=input_dim,
                     hidden_dim=hidden_dim,
                     num_classes=output_dim,
                     Conv1=args.Conv1,
                     Conv2=args.Conv2,
                     image_feature=50176,
                     use_image_feats=args.use_image_feats).to(device)
    
    model= DDP(model,device_ids=[args.gpu_idx],output_device=args.gpu_idx)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=args.wd)
    pbar = tqdm(num_epochs)
    pbar.set_description("training model")
    best_loss=99999
    train_losses = []
    train_accuracies = []
    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer, criterion,device=device)
        train_losses.append(loss)
        if loss <= best_loss:
            best_loss = loss
            pbar.set_description(f"Training model.|Best loss={round(best_loss, 5)}")
        pbar.write(f'Epoch [{epoch}/{num_epochs}]: Loss: {round(loss, 5)}')
        pbar.update(1)
    print(f"Time taken to train the model: { datetime.now() - start_time}")

    plot_and_save_training_performance(num_epochs=num_epochs,
                                       losses=train_losses,
                                       folder_name=config['param']['result_folder'])

    cls_report = test(model=model,
                        loader=test_loader,
                        device=device,
                        class_names=class_names)

    cr = pd.DataFrame(cls_report).transpose()
    cr.to_excel( f"{config['param']['result_folder']}/result_for_GNN_Model.xlsx")
    
    print(f"Model Classification report for GNN model \n ")
    print(cr)
    


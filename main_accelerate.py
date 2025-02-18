import argparse
from datetime import datetime
from torch_geometric.nn import GENConv, GATConv
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from model import *
import os
import gc
from accelerate import Accelerator, InitProcessGroupKwargs

   
if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser()

    parser.add_argument("--type_graph", default="grid", help="define how to construct nodes and egdes", choices=["harris", "grid", "multi"])
    parser.add_argument("--use_image_feats", default=False, type=bool, help="use input  image features as graph feature or not")
    parser.add_argument("--hidden_dim", default=64, type=int, help="hidden_dim")
    parser.add_argument("--num_epochs", type=int, default=2, help="num_epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning_rate")
    parser.add_argument("--wd", type=float, default=0.005, help="wd")
    parser.add_argument("--Conv1", default=GENConv, help="Conv1")
    parser.add_argument("--Conv2", default=GATConv, help="Conv2")
    parser.add_argument("--gpu_idx", default=0, help="GPU  num")
    parser.add_argument("--connectivity", type=str, default="4-connectivity", help="connectivity", choices=["4-connectivity", "8-connectivity"])
    
    args = parser.parse_args()
    
    start_time=datetime.now()

    create_config_file(args.type_graph, args.connectivity)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(backend="gloo")])
    
    train_graph_list,feat_size,class_names = Load_graphdata(f"{config['param']['graph_dataset_folder']}/train")
    test_graph_list,_,_ = Load_graphdata(f"{config['param']['graph_dataset_folder']}/test")

    train_loader= graphdata_loader(train_graph_list,args=args,type_data="train")
    test_loader=graphdata_loader(test_graph_list,args=args,type_data="test")

    
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


    for epoch in range(num_epochs):
        loss = train_function(model=model,
                              dataloader=train_loader, 
                              optimizer=optimizer, 
                              criterion=criterion,
                              accelerator=accelerator)
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

    cls_report = test_function(accelerator=accelerator,
                               model=model,
                               test_loader=test_loader,
                               class_names=class_names)

    cr = pd.DataFrame(cls_report).transpose()
    cr.to_excel( f"{config['param']['result_folder']}/result_for_GNN_Model.xlsx")
    
    print(f"Model Classification report for GNN model \n ")
    print(cr)
    


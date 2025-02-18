import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from datautils import MyTrainDataset
import argparse
from datetime import datetime
from torch_geometric.nn import GENConv, GATConv
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import gc
from model import *



def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="gloo")

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        accumulation_steps: int = 2,
       
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.train_losses = []
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.train_losses = snapshot["TRAIN_LOSSES"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
    
    def _save_snapshot(self, epoch,train_losses):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "TRAIN_LOSSES": train_losses,
            
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int, accumulation_steps: int = 4):
        train_losses = self.train_losses
        for epoch in range(self.epochs_run, max_epochs):
            total_loss = 0
            b_sz = len(next(iter(self.train_data))[0])
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
            self.train_data.sampler.set_epoch(epoch)
            
            # Initialize gradient accumulation counter
            accumulation_counter = 0
            
            for data in self.train_data:
                source = data.to(self.gpu_id)
                targets = data.y.to(self.gpu_id)
                self.optimizer.zero_grad()  # Zero out the gradients before backpropagation
                
                output = self.model(source)
                loss = F.cross_entropy(output, targets)
                loss.backward()  # Backpropagate the loss
                
                # Accumulate gradients
                accumulation_counter += 1
                
                # Perform the optimizer step every `accumulation_steps` batches
                if accumulation_counter % accumulation_steps == 0:
                    self.optimizer.step()  # Update model parameters
                    self.optimizer.zero_grad()  # Zero out gradients after the optimizer step
                    torch.cuda.empty_cache()
                    accumulation_counter = 0  # Reset counter for next accumulation
                
                total_loss += loss.item()  # Track total loss

            # If there are remaining accumulated gradients, perform an update
            if accumulation_counter > 0:
                self.optimizer.step()  # Perform final optimizer step
                self.optimizer.zero_grad()  # Zero out gradients after final step
                torch.cuda.empty_cache()

            # Track average loss for the epoch
            train_losses.append(total_loss / len(self.train_data))
            
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch,train_losses)
                
        return train_losses


def prepare_dataloader(dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )



if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser(description='DP-based GNN training')
    parser.add_argument('--save_every', default=1,type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=4, type=int, help='Input batch size on each device (default: 16)')
    parser.add_argument("--type_graph", default="grid", help="define how to construct nodes and egdes", choices=["harris", "grid", "multi"])
    parser.add_argument("--use_image_feats", default=False, type=bool, help="use input  image features as graph feature or not")
    parser.add_argument("--hidden_dim", default=4, type=int, help="hidden_dim")
    parser.add_argument("--total_epochs", type=int, default=2, help="num_epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning_rate")
    parser.add_argument("--wd", type=float, default=0.005, help="wd")
    parser.add_argument("--Conv1", default=GENConv, help="Conv1")
    parser.add_argument("--Conv2", default=GATConv, help="Conv2")
    parser.add_argument("--accumulation_steps", default=2, help="accumulation_steps")
    parser.add_argument("--connectivity", type=str, default="4-connectivity", help="connectivity", choices=["4-connectivity", "8-connectivity"])
    
    args = parser.parse_args()
    create_config_file(args.type_graph, args.connectivity)

    train_graph_list,feat_size,class_names = Load_graphdata(f"{config['param']['graph_dataset_folder']}/train")
    test_graph_list,_,_ = Load_graphdata(f"{config['param']['graph_dataset_folder']}/test")
    
    gc.collect()
    torch.cuda.empty_cache()
     
    ddp_setup()
    # device = torch.device(f'cuda:{int(os.environ["LOCAL_RANK"])}' if torch.cuda.is_available() else 'cpu')


    start_time=datetime.now()
    train_loader= graphdata_loader(train_graph_list,args=args,type_data="train")
    test_loader=graphdata_loader(test_graph_list,args=args,type_data="test")

    input_dim = feat_size
    hidden_dim = args.hidden_dim
    output_dim = 10
    num_epochs = args.total_epochs
    batch_size = args.batch_size
    
    
    save_every=args.save_every
    total_epochs=args.total_epochs
    batch_size=args.batch_size
    
    model = GNNModel(num_node_features=input_dim,
                     hidden_dim=hidden_dim,
                     num_classes=output_dim,
                     Conv1=args.Conv1,
                     Conv2=args.Conv2,
                     image_feature=50176,
                     use_image_feats=args.use_image_feats)
    

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    trainer = Trainer(model, train_loader, optimizer, save_every, snapshot_path="snapshot.pth",accumulation_steps = args.accumulation_steps)
    train_losses = trainer.train(total_epochs)
    
    plot_and_save_training_performance(num_epochs=num_epochs,
                                       losses=train_losses,
                                       folder_name=config['param']['result_folder'])

    cls_report = test_ddp(model=model,
                        loader=test_loader,
                        class_names=class_names)
    destroy_process_group()
    cr = pd.DataFrame(cls_report).transpose()
    cr.to_excel( f"{config['param']['result_folder']}/result_for_GNN_Model.xlsx")
    
    print(f"Model Classification report for GNN model \n ")
    print(cr)
    

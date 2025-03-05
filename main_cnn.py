import argparse
import pandas as pd
import os
from accelerate import Accelerator, InitProcessGroupKwargs
import torch.nn as nn
from model import CNNModel
import torch
import torch.optim as optim
from Baselines.baseline_models import baseline_model
from train_test_model import train_model, test_model
from datetime import datetime
from utils import *


if __name__ == "__main__":
    set_seed()
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(backend="gloo")])
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", help="dataset name", default="lidl", choices=["lidl", "other"])
    parser.add_argument("--type_model", help="type of the model Baseline or our own CNN model", default="baseline", choices=["baseline", "Our_CNN_Model"])
    parser.add_argument("--model_name", help="Model name", default="ResNet50", choices=["VGG19", "VGG16", "ResNet50",  "ResNet101","AlexNet", "MobileNetV2", "GoogleNet"])
    parser.add_argument("--dataset_size", type=int, default=0, help="number  of images to use for training per class, 0 means all")
    parser.add_argument("--use_class_weights", default=True, type=bool, help="use class weights", choices=[True, False])
    parser.add_argument("--hidden_dim", default=512, type=int, help="hidden_dim")
    parser.add_argument("--num_epochs", type=int, default=150, help="num_epochs")
    parser.add_argument("--batch_size", type=int, default=32*4, help="batch_size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning_rate")
    parser.add_argument("--wd", type=float, default=0.005, help="wd")
    parser.add_argument("--criterion", default="CrossEntropy", help="criterion")
    parser.add_argument("--gpu_idx", default=1, help="GPU  num")

    args = parser.parse_args()
    train_data= "dataset/images/train"
    val_data = "dataset/images/val"
    test_data = "dataset/images/test"
    num_classes, train_loader,val_loader, test_loader, class_names, sample_weights = load_data(dataset_dir=[train_data,val_data,test_data], batch_size=args.batch_size, num_samples_per_class=args.dataset_size, use_class_weights=args.use_class_weights)

    start_time = datetime.now()
    if args.type_model == "baseline":
        model = baseline_model(model_name=args.model_name, num_classes=num_classes)
    elif args.type_model == "Our_CNN_Model":
        model = CNNModel()
        args.model_name = "New_CNN_Model"
        
    print(f"Model: {args.model_name} | device: {accelerator.device}")

    if args.dataset_size == 0:
        args.result_dir = f"results/CNN Models/full_dataset/{args.model_name}"
    else:
        args.result_dir = f"results/CNN Models/{args.dataset_size}images_per_class/{args.model_name}"
    os.makedirs(args.result_dir, exist_ok=True)
    saved_model_path = f'{args.result_dir}/{args.model_name}_weight.pth'
    
    if os.path.isfile(saved_model_path):
        try:
            model.load_state_dict(torch.load(saved_model_path))
            print(f"Loaded saved model from {saved_model_path}")
        except:
            pass
    print(f" Sample weights: {len(sample_weights)}: {sample_weights}")
    criterion = nn.CrossEntropyLoss(reduction='mean')
        # criterion = nn.CrossEntropyLoss(weight=torch.tensor(sample_weights).to(accelerator.device),reduction='mean')

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    
    
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader,test_loader)

    model=train_model(model,accelerator, train_loader, criterion, optimizer, args=args)
    # torch.save(model.state_dict(), saved_model_path)
    end_time = datetime.now()
    cl_report = test_model(model, accelerator,test_loader, class_names,args=args)

    cr = pd.DataFrame(cl_report).transpose()
    cr.to_excel(f"{args.result_dir}/result_for_{args.model_name}.xlsx")

    print(f"Model Classification report for {args.model_name} \n ")
    print(cr)
    times=calculate_running_time(start_time,end_time)
    print(f"Time taken to train the model: {times} {'hours' if times>1 else 'hour'}")
import argparse
import pandas as pd
import os

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
    parser = argparse.ArgumentParser()

    parser.add_argument("--type_model", help="type of the model Baseline or our own CNN model", default="Our_CNN_Model", choices=["baseline", "Our_CNN_Model"])
    parser.add_argument("--model_name", help="Model name", default="MobileNetV2", choices=["VGG19", "VGG16", "ResNet50", "AlexNet", "MobileNetV2", "GoogleNet"])
    parser.add_argument("--dataset_size", type=int, default=0, help="number  of images to use for training per class, 0 means all")
    parser.add_argument("--hidden_dim", default=256, type=int, help="hidden_dim")
    parser.add_argument("--num_epochs", type=int, default=100, help="num_epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning_rate")
    parser.add_argument("--wd", type=float, default=0.001, help="wd")
    parser.add_argument("--criterion", default="CrossEntropy", help="criterion")
    parser.add_argument("--gpu_idx", default=2, help="GPU  num")

    args = parser.parse_args()

    args.device = torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')
    print("device:", args.device)
    num_classes, train_loader, class_names = load_data(dataset_dir="dataset/images/train", batch_size=args.batch_size, num_samples_per_class=args.dataset_size,type_data="train")
    _, test_loader, _ = load_data(dataset_dir="dataset/images/test", batch_size=args.batch_size,num_samples_per_class=args.dataset_size,type_data="test")


    start_time = datetime.now()
    if args.type_model == "baseline":
        model = baseline_model(model_name=args.model_name, num_classes=num_classes)
    elif args.type_model == "Our_CNN_Model":
        model = CNNModel()
        args.model_name = "New_CNN_Model"

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
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    model= train_model(model, train_loader, test_loader, criterion, optimizer, args=args)
    torch.save(model.state_dict(), saved_model_path)
    end_time = datetime.now()
    cl_report = test_model(model, test_loader, class_names,args=args)

    cr = pd.DataFrame(cl_report).transpose()
    cr.to_excel(f"{args.result_dir}/result_for_{args.model_name}.xlsx")

    print(f"Model Classification report for {args.model_name} \n ")
    print(cr)
    print(f"Time taken to train the model: {end_time - start_time}")

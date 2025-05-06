import argparse
import pandas as pd
import torch.nn as nn
#from ..model import CNNModel
from utils import *
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", torch.cuda.is_available())
from baseline_models import  baseline_model
from train_test_model import train_model, test_model
from datetime import datetime


if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--type_model", help="type of the model Baseline or our own CNN model", default="baseline", choices=["baseline", "ours"])
    parser.add_argument("--model_name", help="Model name", default="ResNet50", choices=["VGG19", "VGG16","ResNet50","AlexNet", "MobileNetV2","Our_CNN_Model"])
    parser.add_argument("--hidden_dim", default=128, type=int, help="hidden_dim")
    parser.add_argument("--num_epochs", type=int, default=50, help="num_epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="learning_rate")
    parser.add_argument("--wd", type=float, default=0.0001, help="wd")
    parser.add_argument("--criterion", default="CrossEntropy", help="criterion")

    args = parser.parse_args()
    args.result_dir = f"results/{args.model_name}"
    os.makedirs(args.result_dir, exist_ok=True)
    
    num_classes, train_loader, class_names = load_data(dataset_dir="dataset/images/train", batch_size=args.batch_size)
    _, test_loader, _ = load_data(dataset_dir="dataset/images/test", batch_size=args.batch_size)
    
    if args.type_model=="baseline":
                 model = baseline_model(model_name=args.model_name,num_classes=num_classes)
    elif args.type_model=="Our_CNN_Model":
        model= CNNModel()
        args.model_name="Our_CNN_Model"

    saved_model_path = f'{args.model_name}_weight.pth'
    if os.path.isfile(saved_model_path):
        try:
            model.load_state_dict(torch.load(saved_model_path))
            print(f"Loaded saved model from {saved_model_path}")
        except:
            pass
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    train_model(model, train_loader, test_loader, criterion, optimizer, args.num_epochs,result_dir=args.result_dir)
    cl_report = test_model(model, test_loader,class_names, args)

    cr = pd.DataFrame(cl_report).transpose()

    with open(f"{args.result_dir}/result_for_{args.model_name}.txt", 'a') as result_file:
        result_file.write(f"{'+' * 12}Report time: {datetime.now().strftime('%Y-%m-%d %H:%M')} {'+' * 12}\n")
        result_file.write(cr)
        result_file.write("==" * 25 + "\n")

    print("Model Classification report \n ")
    print(cr)
import argparse
import pandas as pd
import os
from accelerate import Accelerator, InitProcessGroupKwargs
import torch.nn as nn
from model import CNNModel,HybridImageClassifier
import torch
import torch.optim.lr_scheduler as lr_scheduler

import torch.optim as optim
from Baselines.baseline_models import baseline_model
from train_test_model import train_model, test_model,train_hybrid_model,test_hybrid_model
from datetime import datetime
from utils import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":
    set_seed()
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(backend="gloo")])
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", help="dataset name", default="lidl", choices=["lidl", "other"])
    parser.add_argument("--type_model", help="type of the model Baseline or our own CNN model", default="baseline", choices=["baseline", "Our_CNN_Model","hybrid"])
    parser.add_argument("--model_name", help="Model name", default="VGG19", choices=["VGG19", "VGG16", "ResNet50",  "ResNet101","AlexNet", "MobileNetV2", "GoogleNet","Unet"])
    parser.add_argument("--dataset_size", type=int, default=0, help="number  of images to use for training per class, 0 means all")
    parser.add_argument("--use_class_weights", default=True, type=bool, help="use class weights", choices=[True, False])
    parser.add_argument("--hidden_dim", default=128, type=int, help="hidden_dim")
    parser.add_argument("--add_fix_feats", default=240, type=int, help="addiional fixed feature size per view")
    parser.add_argument("--num_epochs", type=int, default=200, help="num_epochs")
    parser.add_argument("--batch_size", type=int, default=32*4, help="batch_size")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning_rate")
    parser.add_argument("--wd", type=float, default=0.0001, help="wd")
    parser.add_argument("--criterion", default="CrossEntropy", help="criterion")
    parser.add_argument("--gpu_idx", default=4, help="GPU  num")

    args = parser.parse_args()                    
    train_data= "dataset/images/train"
    val_data = "dataset/images/val"
    test_data = "dataset/images/test"
    # feature_list=["grid_features","keypoint_features","feature_map_features","superpixel_features","region_adjacency_features","mesh3d_features"]
    feature_list=["superpixel_features","region_adjacency_features"]
    print(f"{'*'*10} Using {args.type_model} model with {args.model_name} architecture  {'*'*10} \n")
    if args.type_model != "hybrid":
        num_classes, train_loader,val_loader, test_loader, class_names, class_weights = load_data(dataset_dir=[train_data,val_data,test_data], batch_size=args.batch_size, num_samples_per_class=args.dataset_size, use_class_weights=args.use_class_weights)
    elif args.type_model == "hybrid":
        train_loader, label_encoder,class_names = create_dataloader(data_dir=train_data,
                                                        feature_list= feature_list,
                                                        fixed_size=args.add_fix_feats,
                                                        batch_size=args.batch_size,
                                                        sufle=True)
        val_loader, _,_ = create_dataloader(data_dir=val_data,
                                                        feature_list= feature_list,
                                                        fixed_size=args.add_fix_feats,
                                                        batch_size=args.batch_size,
                                                        sufle=False)
        test_loader, _,_ = create_dataloader(data_dir=test_data,
                                                        feature_list= feature_list,
                                                        fixed_size=args.add_fix_feats,
                                                        batch_size=args.batch_size,
                                                        sufle=False)
        
        num_classes = len(label_encoder.classes_)
        class_weights = compute_class_weights(train_loader, num_classes, accelerator.device)

        
        sample_image, sample_features, _ = next(iter(train_loader))
        feature_size = sample_features.shape[1]
        print(f"Additional Feature size is {feature_size}")  
        print(f"Image Feature size is {sample_image.shape}")  
            
    start_time = datetime.now()
    if args.type_model == "baseline":
        model = baseline_model(model_name=args.model_name, num_classes=num_classes)
    elif args.type_model == "Our_CNN_Model":
        model = CNNModel()
        args.model_name = "New_CNN_Model"
    elif args.type_model=="hybrid":
        model=HybridImageClassifier(num_classes=num_classes,feature_size=feature_size)
        args.model_name = "Hybrid_Model"
        
    # print(f"Model: {args.model_name} | device: {accelerator.device}")
    # for name, param in model.named_parameters():
    #     print(f"{name}: Requires Grad = {param.requires_grad}")


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
    criterion = nn.CrossEntropyLoss(reduction='mean')
    #criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    # criterion= FocalLoss(alpha=class_weights,gamma=2.0, reduction="mean")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_loader), epochs=args.num_epochs)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters in the model: {pytorch_total_params}")
    
    model, optimizer,scheduler, train_loader, val_loader,test_loader = accelerator.prepare(model, optimizer,scheduler, train_loader,val_loader,test_loader)

    if args.type_model=="hybrid":
         model=train_hybrid_model(model,accelerator, train_loader, val_loader,criterion, optimizer, scheduler,args=args)
    else:
        model=train_model(model,accelerator, train_loader, val_loader,criterion, optimizer,scheduler, args=args)
        
    # torch.save(model.state_dict(), saved_model_path)
    end_time = datetime.now()
    times=round((end_time - start_time).total_seconds(),2)
     
    if args.type_model=="hybrid":
        cl_report = test_hybrid_model(model, accelerator,test_loader, class_names,args=args)
    else:
        cl_report = test_model(model, accelerator,test_loader, class_names,args=args)
   
    cr = pd.DataFrame(cl_report).transpose()
    cr.to_excel(f"{args.result_dir}/result_for_{args.model_name}_param_{pytorch_total_params}_({args.gpu_idx})GPU_{times}_seconds.xlsx")

    print(f"Model Classification report for {args.model_name} \n ")
    print(cr)
        
    print(f"Time taken to train the {args.model_name} model with {pytorch_total_params} parmaters using {args.gpu_idx} GPU: {times/3600} {'hours' if times>1 else 'hour'}")
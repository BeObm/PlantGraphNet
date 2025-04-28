import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from model import CNNModel
from utils import *
import os
from tqdm import tqdm



def train_model(model,accelerator, train_loader, val_loader, criterion, optimizer,scheduler, args):
    # device = args.device
    # Lists to store loss and accuracy values for plotting
    train_loss_values = []
    train_accuracy_values = []
    test_accuracy_values_during_training = []
    print("Trainning Model...")
    model.train()
    for epoch in tqdm(range(args.num_epochs)):
       
        running_loss = 0.0

        for inputs, labels in train_loader:
            # inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            running_loss += accelerator.gather(loss).sum().item() * inputs.size(0)
            optimizer.step()
            scheduler.step()

        model.eval()

        # Compute and store training loss and accuracy
        train_loss = running_loss / len(train_loader.dataset)
        train_loss_values.append(train_loss)
        total = 0
        correct = 0


        with torch.no_grad():
            for inputs, labels in val_loader:
                # inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                #print(f"Model Outputs: {outputs[:5]}")  # Print first 5 predictions
                #print(f"True Labels: {labels[:5]}")
                _, predicted = torch.max(outputs, 1)
                all_predict=accelerator.gather(predicted)
                all_labels=accelerator.gather(labels)
                total += all_labels.size(0)
                correct += (all_predict == all_labels).sum().item()

        train_accuracy = 100 * correct / total
        train_accuracy_values.append(train_accuracy)

       
      
        print(f' <<{"="*8}  Epoch {epoch + 1}| Train Loss: {train_loss} | Train Accuracy: {train_accuracy}%   {"="*8}>> \n ')


    # Save training stats
    plot_and_save_training_performance(num_epochs=args.num_epochs,
                                       losses=train_loss_values,
                                       folder_name=args.result_dir)

    return model

def test_model(model, accelerator, test_loader, class_names,args):
    # device = args.device
    # model = model.to(device)    
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            # inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predict=accelerator.gather(predicted)
            all_labels=accelerator.gather(labels)   
            y_pred.append(all_predict.cpu())
            y_true.append(all_labels.cpu())

    # Concatenate the list of tensors into a single tensor for y_pred and y_true
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)

    plot_confusion_matrix(y_true=y_true,
                          y_pred=y_pred,
                          class_names=class_names,
                          file_name= f"{args.result_dir}/confusion_matrix.pdf"
                          )
    print(f"Confusion Matrix for {args.model_name} is saved in {args.model_name}_confusion_matrix.pdf")

    cls_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    return cls_report



# Function to train the model
def train_hybrid_model(model,accelerator, train_loader, val_loader,criterion, optimizer,scheduler, args):
    train_loss_values = []
    train_accuracy_values = []

    for epoch in tqdm(range(args.num_epochs)):
        
        # Training Phase
        model.train()
        running_loss = 0.0
        correct=0
        total=0
        
        for images, features, labels in train_loader:
            
            optimizer.zero_grad()
            outputs = model(images, features)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            running_loss += accelerator.gather(loss).sum().item() * images.size(0)

            logits = torch.argmax(outputs, dim=1)
            
            all_preds=accelerator.gather(logits)
            all_labels = accelerator.gather(labels)
            
            correct += (all_preds == all_labels).sum().item()
            total += all_labels.size(0)

        train_acc = correct / total
        avg_train_loss = running_loss / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        
        train_loss_values.append(avg_train_loss)
        total = 0
        correct = 0
        
        with torch.no_grad():
            for images, features, labels in val_loader:
                outputs = model(images, features)
                # print(f"Model Outputs: {outputs[:5]}")  # Print first 5 predictions
                # print(f"True Labels: {labels[:5]}")
                predicted = torch.argmax(outputs, dim=1)
                all_predict=accelerator.gather(predicted)
                all_labels=accelerator.gather(labels)
                total += all_labels.size(0)
                correct += (all_predict == all_labels).sum().item()

        train_accuracy = correct / total
        train_accuracy_values.append(train_accuracy)


        # Print progress
        print(f"\n Epoch {epoch+1}/{args.num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f" Val Acc: {train_accuracy:.4f} "
             )
    plot_and_save_training_performance(num_epochs=args.num_epochs,
                                       losses=train_loss_values,
                                       folder_name=args.result_dir)

    return model






def test_hybrid_model(model, accelerator, test_loader, class_names,args):
    # device = args.device
    # model = model.to(device)    
    model.eval()
    y_pred = []
    y_true = []


    with torch.no_grad():
            for images, features, labels in test_loader:
                outputs = model(images, features)
                
                predicted = torch.argmax(outputs, dim=1)
                all_predict=accelerator.gather(predicted)
                all_labels=accelerator.gather(labels)
                y_pred.append(all_predict.cpu())
                y_true.append(all_labels.cpu())

    # Concatenate the list of tensors into a single tensor for y_pred and y_true
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)

    plot_confusion_matrix(y_true=y_true,
                          y_pred=y_pred,
                          class_names=class_names,
                          file_name= f"{args.result_dir}/confusion_matrix.pdf"
                          )
    print(f"Confusion Matrix for {args.model_name} is saved in {args.model_name}_confusion_matrix.pdf")

    cls_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    return cls_report

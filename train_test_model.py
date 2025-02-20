import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from model import CNNModel
from utils import *
import os
from tqdm import tqdm



def train_model(model,accelerator, train_loader, criterion, optimizer, args):
    # device = args.device
    # Lists to store loss and accuracy values for plotting
    train_loss_values = []
    train_accuracy_values = []
    test_accuracy_values_during_training = []
    print("Trainning Model...")

    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            # inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            running_loss += float(loss.item())*inputs.size(0)
            optimizer.step()

        model.eval()

        # Compute and store training loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_loss_values.append(train_loss)
        total = 0
        correct = 0


        with torch.no_grad():
            for inputs, labels in train_loader:
                # inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_predict=accelerator.gather(predicted)
                all_labels=accelerator.gather(labels)
                total += all_labels.size(0)
                correct += (all_predict == all_labels).sum().item()

        train_accuracy = 100 * correct / total
        train_accuracy_values.append(train_accuracy)

        # total = 0
        # correct = 0
        # with torch.no_grad():
        #     for inputs, labels in test_loader:
        #         inputs, labels = inputs.to(device), labels.to(device)
        #         outputs = model(inputs)
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
        # test_accuracy = 100 * correct / total
        # test_accuracy_values_during_training.append(test_accuracy)
        # print(f"total: {total} correct: {correct} test_accuracy: {test_accuracy} train_accuracy: {train_accuracy} train_loss: {train_loss} \n")
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



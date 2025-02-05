import torch.nn as nn
import torchvision.models as models



def baseline_model(model_name, num_classes):
    if model_name == "AlexNet":
        return AlexNet_Model(num_classes)
    elif model_name == "MobileNetV2":
        return MobileNetV2_Model(num_classes)
    elif model_name == "ResNet50":
        return ResNet50_Model(num_classes)
    elif model_name == "ResNet101":
        return ResNet101_Model(num_classes)
    elif model_name == "VGG16":
        return VGG16_Model(num_classes)
    elif model_name == "VGG19":
        return VGG19_Model(num_classes)

    elif model_name == "GoogleNet":
        return GoogleNet_Model(num_classes)





def AlexNet_Model(num_classes):
    model = models.alexnet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    model.classifier[6] = nn.Sequential(
        nn.Linear(model.classifier[6].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model


def MobileNetV2_Model(num_classes):
    model = models.mobilenet_v2(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    model.classifier[1] = nn.Sequential(
        nn.Linear(model.last_channel, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model


def ResNet50_Model(num_classes):
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model



def ResNet101_Model(num_classes):
    model = models.resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

def VGG16_Model(num_classes):
    model = models.vgg16(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False  # Freeze convolutional layers

    model.classifier[6] = nn.Sequential(
        nn.Linear(model.classifier[6].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model


def VGG19_Model(num_classes):
    model = models.vgg19(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False  # Freeze convolutional layers

    model.classifier[6] = nn.Sequential(
        nn.Linear(model.classifier[6].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model


def GoogleNet_Model(num_classes):
    model = models.googlenet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

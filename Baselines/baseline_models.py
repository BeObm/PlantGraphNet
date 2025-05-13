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
    elif model_name == "Unet":
        return UNetClassifier(num_classes=num_classes)




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


class UNetClassifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(UNetClassifier, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(256, 512)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(self.pool1(x))
        x = self.encoder3(self.pool2(x))
        x = self.bottleneck(self.pool3(x))
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

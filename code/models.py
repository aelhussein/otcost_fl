
from configs import ROOT_DIR
import torch.nn.functional as F
import torch.nn as nn
import torch
from unet import UNet
from torchvision.models import resnet18


class L2Normalize(torch.nn.Module):
    def __init__(self, dim=1):
        super(L2Normalize, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)
    

class Synthetic(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(Synthetic, self).__init__()
        self.input_size = 15
        self.hidden_size = 15
        
        self.fc = torch.nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, 2)
        )
        
        self.sigmoid = torch.nn.Sigmoid()
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        x = x.squeeze(1)
        return self.fc(x)


class Heart(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(Heart, self).__init__()
        self.input_size = 10
        self.hidden_size = 10
        
        self.fc = torch.nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, 5)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        x = x.squeeze(1)
        if x.dtype != torch.float32:
            x = x.float()
        return self.fc(x)

class Credit(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(Credit, self).__init__()
        self.input_size = 29
        self.hidden_size = [10]
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size[0]),
            nn.LayerNorm(self.hidden_size[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size[0], 2)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        x = x.squeeze(1)
        return self.fc(x)


class EMNIST(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(EMNIST, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(256, 120)
        self.norm1 = nn.LayerNorm(120)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(120, 20)
        self.norm2 = nn.LayerNorm(20)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(20, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        
        out = self.fc(out)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.fc1(out)
        out = self.norm2(out)
        out = self.relu1(out)
        out = self.dropout2(out)
        
        out = self.fc2(out)
        return out

# class CIFAR(nn.Module):
#     def __init__(self):
#         super(CIFAR, self).__init__()
        
#         self.resnet = resnet18(weights='ResNet18_Weights.DEFAULT')
#         for param in self.resnet.parameters():
#             param.requires_grad = False
#         for param in self.resnet.layer4.parameters():
#             param.requires_grad = True
        
#         num_ftrs = self.resnet.fc.in_features
#         self.resnet.fc = nn.Sequential(nn.Linear(num_ftrs, 20),
#                                         nn.ReLU(),
#                                         nn.Linear(20, 10)
#         )
#         for layer in self.resnet.fc:
#                 if isinstance(layer, nn.Linear):
#                         nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
#                         nn.init.constant_(layer.bias, 0)
#     def forward(self, x):
#         x = self.resnet(x)
#         return x
    

class CIFAR(nn.Module):
    def __init__(self, dropout_rate=0.25):
        super(CIFAR, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # Input: 3x32x32
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 32x16x16
        self.dropout_conv1 = nn.Dropout2d(p=dropout_rate * 0.5) # Optional: Spatial Dropout

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Input: 32x16x16
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 64x8x8
        self.dropout_conv2 = nn.Dropout2d(p=dropout_rate * 0.5) # Optional: Spatial Dropout

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Input: 64x8x8
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 128x4x4

        # Fixed flattened size for 32x32 input
        self.flattened_size = 128 * 4 * 4

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        # LayerNorm is often suitable for FC layers in FL
        self.ln1 = nn.LayerNorm(256)
        self.relu4 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(p=dropout_rate)

        self.fc2 = nn.Linear(256, 10) # Output layer for 10 CIFAR-10 classes

    def forward(self, x):
        # Conv Block 1
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.dropout_conv1(out) # Apply dropout after pooling

        # Conv Block 2
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.dropout_conv2(out) # Apply dropout after pooling

        # Conv Block 3
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        # Flatten
        out = out.view(out.size(0), -1) # Flatten the feature map

        # FC Block 1
        out = self.fc1(out)
        out = self.ln1(out) # Use LayerNorm here
        out = self.relu4(out)
        out = self.dropout_fc1(out)

        # Output Layer
        out = self.fc2(out)
        return out
    
class IXITiny(nn.Module):
    def __init__(self):
        super(IXITiny, self).__init__()
        self.CHANNELS_DIMENSION = 1
        self.SPATIAL_DIMENSIONS = (2, 3, 4) 

        self.model = UNet(
            in_channels=1,
            out_classes=2,
            dimensions=3,
            num_encoding_blocks=3,
            out_channels_first_layer=8,
            normalization='batch',
            upsampling_type='linear',
            padding=True,
            activation='PReLU',
        )
        checkpoint = torch.load(
            f'{ROOT_DIR}/data/IXITiny/whole_images_epoch_5.pth', 
            map_location=torch.device('cpu')
        )
        self.model.load_state_dict(checkpoint['weights'])

        # Enable gradient computation for all parameters
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        logits = self.model(x)
        return F.softmax(logits, dim=self.CHANNELS_DIMENSION)
    
    def initialize_weights(self):
        # This method needs classifier attribute to work
        if hasattr(self, 'classifier') and isinstance(self.classifier, nn.Conv3d):
            nn.init.xavier_normal_(self.classifier.weight.data)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias.data, 0)

class ISIC(nn.Module):
    def __init__(self):
        super(ISIC, self).__init__()
        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        for _, param in self.efficientnet.named_parameters():
            param.requires_grad = True
        nftrs = self.efficientnet.classifier.fc.in_features 
        self.efficientnet.classifier.fc = nn.Linear(nftrs, 8)

        self.initialize_weights()

    def forward(self, x):
        logits = self.efficientnet(x)
        return logits

    def initialize_weights(self):
        if hasattr(self.efficientnet, 'classifier'):
            nn.init.xavier_normal_(self.efficientnet.classifier.fc.weight.data)
            nn.init.constant_(self.efficientnet.classifier.fc.bias.data, 0)


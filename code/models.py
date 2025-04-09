
from configs import *

class Synthetic(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(Synthetic, self).__init__()
        self.input_size = 12
        self.hidden_size = 12
        
        self.fc = torch.nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, 1)
        )
        
        self.sigmoid = torch.nn.Sigmoid()
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        x = x.squeeze(1)
        return self.sigmoid(self.fc(x))

class Credit(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(Credit, self).__init__()
        self.input_size = 28
        self.hidden_size = [56, 56]
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size[0]),
            nn.BatchNorm1d(self.hidden_size[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.BatchNorm1d(self.hidden_size[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size[1], 1)
        )
        
        self.sigmoid = torch.nn.Sigmoid()
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        x = x.squeeze(1)
        return self.sigmoid(self.fc(x))

class Weather(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(Weather, self).__init__()
        self.input_size = 123
        self.hidden_size = [123, 123, 50]
        
        self.fc = nn.Sequential(
            # First block
            nn.Linear(self.input_size, self.hidden_size[0]),
            nn.BatchNorm1d(self.hidden_size[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # # Second block
            # nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            # nn.BatchNorm1d(self.hidden_size[1]),
            # nn.ReLU(),
            # nn.Dropout(dropout_rate),
            # # Third block
            # nn.Linear(self.hidden_size[1], self.hidden_size[2]),
            # nn.BatchNorm1d(self.hidden_size[2]),
            # nn.ReLU(),
            # Output layer
            nn.Linear(self.hidden_size[0], 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        x = x.squeeze(1)
        return self.fc(x)

class EMNIST(nn.Module):
    def __init__(self, CLASSES):
        super(EMNIST, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(256, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, CLASSES)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

class CIFAR(nn.Module):
    def __init__(self, CLASSES):
        super(CIFAR, self).__init__()
        
        self.resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Linear(num_ftrs, 200),
                                        nn.ReLU(),
                                        nn.Linear(200, CLASSES)
        )
        for layer in self.resnet.fc:
                if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                        nn.init.constant_(layer.bias, 0)
    def forward(self, x):
        x = self.resnet(x)
        return x
    
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


global ROOT_DIR
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl'
global DATA_DIR
DATA_DIR = f'{ROOT_DIR}/data/ISIC'

import sys
import os
sys.path.append(f'{ROOT_DIR}/code/OTcost/')
import numpy as np
import pandas as pd
import ISICdataset
import importlib
importlib.reload(ISICdataset)
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader as dl
from collections import OrderedDict
from multiprocessing import Pool
import matplotlib.pyplot as plt

BATCH_SIZE = 8


import torch
import torch.nn as nn

def get_configs(arch='vgg16'):

    if arch == 'vgg11':
        configs = [1, 1, 2, 2, 2]
    elif arch == 'vgg13':
        configs = [2, 2, 2, 2, 2]
    elif arch == 'vgg16':
        configs = [2, 2, 3, 3, 3]
    elif arch == 'vgg19':
        configs = [2, 2, 4, 4, 4]
    else:
        raise ValueError("Undefined model")
    
    return configs

class VGGAutoEncoder(nn.Module):

    def __init__(self, configs, n_emb):

        super(VGGAutoEncoder, self).__init__()

        # VGG without Bn as AutoEncoder is hard to train
        self.encoder = VGGEncoder(configs=configs, enable_bn=True)
        self.fc = nn.Sequential(
            nn.Linear(in_features=18432, out_features=4096),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=8),
            nn.Sigmoid())
        self.compressor = nn.Sequential(
            nn.Linear(in_features=18432, out_features=4096),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=n_emb),
            nn.ReLU())
        self.decompressor = nn.Sequential(
            nn.Linear(in_features=n_emb, out_features=4096),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=18432),
            nn.ReLU())
        
        self.decoder = VGGDecoder(configs=configs[::-1], enable_bn=True)
        
    
    def freeze_encoder_decoder(self):
        """Freeze the encoder and decoder, keeping only FC, compressor, decompressor trainable"""
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Freeze decoder
        for param in self.decoder.parameters():
            param.requires_grad = False

    def get_trainable_params(self):
        """Return only the parameters that are set to be trained"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def forward(self, x, embedding = False):

        x = self.encoder(x)
        logits = self.fc(x)
        x = self.compressor(x)
        if embedding:
            return x
        x = self.decompressor(x)
        x = self.decoder(x)

        return x, logits

class VGGEncoder(nn.Module):

    def __init__(self, configs, enable_bn=False):

        super(VGGEncoder, self).__init__()

        if len(configs) != 5:

            raise ValueError("There should be 5 stage in VGG")

        self.conv1 = EncoderBlock(input_dim=3,   output_dim=64,  hidden_dim=64,  layers=configs[0], enable_bn=enable_bn)
        self.conv2 = EncoderBlock(input_dim=64,  output_dim=128, hidden_dim=128, layers=configs[1], enable_bn=enable_bn)
        self.conv3 = EncoderBlock(input_dim=128, output_dim=256, hidden_dim=256, layers=configs[2], enable_bn=enable_bn)
        self.conv4 = EncoderBlock(input_dim=256, output_dim=512, hidden_dim=512, layers=configs[3], enable_bn=enable_bn)
        self.conv5 = EncoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=configs[4], enable_bn=enable_bn)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

class VGGDecoder(nn.Module):

    def __init__(self, configs, enable_bn=False):

        super(VGGDecoder, self).__init__()

        if len(configs) != 5:

            raise ValueError("There should be 5 stage in VGG")

        self.conv1 = DecoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=configs[0], enable_bn=enable_bn)
        self.conv2 = DecoderBlock(input_dim=512, output_dim=256, hidden_dim=512, layers=configs[1], enable_bn=enable_bn)
        self.conv3 = DecoderBlock(input_dim=256, output_dim=128, hidden_dim=256, layers=configs[2], enable_bn=enable_bn)
        self.conv4 = DecoderBlock(input_dim=128, output_dim=64,  hidden_dim=128, layers=configs[3], enable_bn=enable_bn)
        self.conv5 = DecoderBlock(input_dim=64,  output_dim=3,   hidden_dim=64,  layers=configs[4], enable_bn=enable_bn)
        self.gate = nn.Sigmoid()
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x

class EncoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(EncoderBlock, self).__init__()

        if layers == 1:

            layer = EncoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('0 EncoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = EncoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                
                self.add_module('%d EncoderLayer' % i, layer)
        
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.add_module('%d MaxPooling' % layers, maxpool)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class DecoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(DecoderBlock, self).__init__()

        upsample = nn.ConvTranspose2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=2, stride=2)

        self.add_module('0 UpSampling', upsample)

        if layers == 1:

            layer = DecoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('1 DecoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = DecoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                
                self.add_module('%d DecoderLayer' % (i+1), layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class EncoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super(EncoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
    
    def forward(self, x):

        return self.layer(x)

class DecoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super(DecoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(input_dim),
                nn.ReLU(),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.layer = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )
    
    def forward(self, x):

        return self.layer(x)


def total_loss(recon_x, x, logits, labels, alpha=0.7, beta=0.3):
    # Pixel-wise reconstruction loss
    mse_loss = F.mse_loss(recon_x, x)
    
    # Structural similarity loss (simplified SSIM)
    # Using a smoother L1 loss on gradient differences to preserve edges
    x_diff_x = x[:, :, 1:, :] - x[:, :, :-1, :]
    x_diff_y = x[:, :, :, 1:] - x[:, :, :, :-1]
    recon_diff_x = recon_x[:, :, 1:, :] - recon_x[:, :, :-1, :]
    recon_diff_y = recon_x[:, :, :, 1:] - recon_x[:, :, :, :-1]
    
    diff_x_loss = F.smooth_l1_loss(recon_diff_x, x_diff_x)
    diff_y_loss = F.smooth_l1_loss(recon_diff_y, x_diff_y)
    structural_loss = (diff_x_loss + diff_y_loss) / 2.0
    
    # Combined reconstruction loss
    recon_loss = alpha * mse_loss + beta * structural_loss
    
    # Classification loss
    cls_loss = F.cross_entropy(logits, labels)
    
    # Combined total loss
    total = recon_loss + cls_loss
    
    return total



def load_model():
    ## Pretrained
    configs = get_configs('vgg16')
    model = VGGAutoEncoder(configs)
    checkpoint = torch.load(f'{DATA_DIR}/imagenet-vgg16.pth', map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if 'module.' in key:
            name = key[7:] # remove 'module.' prefix
        else:
            name = key 
        new_state_dict[name] = value
    model.load_state_dict(new_state_dict)
    return model

def extract_embedding(model, image):
    B = image.shape[0]
    with torch.no_grad():
        image = image.transpose(2,1)
        embedding = model(image, embedding = True)
        return embedding.reshape(B,-1).detach()

def create_embedding(loader, center):
    labels_list = []
    names_list = []
    model = load_model()
    for images, labels, paths in loader:
        embeddings = extract_embedding(model, images)
        image_names = [p.split('/')[-1].split('.')[0] for p in paths]
        names_list.extend(image_names)
        emb_paths = [f'{DATA_DIR}/embedding/center_{center}_{image_name}' for image_name in image_names]
        for i in range(len(emb_paths)):
            emb_save = embeddings[i].numpy()
            emb_path = emb_paths[i]
            np.save(emb_path, emb_save)
        labels_list.append(labels)   
    all_labels = (torch.cat(labels_list, dim=0)).numpy()
    labels_df = pd.DataFrame({
                    "Name": names_list,
                    "Label": all_labels
                })
    labels_df.to_csv(f'{DATA_DIR}/center_{center}_labels.csv', index = False)
    return

def train_vgg_autoencoder(model, train_loader, val_loader, n_emb, device=None, 
                          num_epochs=50, learning_rate=0.001, patience=10, 
                          save_dir=None, center=None):
    """
    Train a VGG autoencoder with frozen encoder/decoder components
    
    Args:
        model: VGGAutoEncoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        n_emb: Embedding dimension (for saving model)
        device: Device to train on ('cuda' or 'cpu')
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate for optimizer
        patience: Early stopping patience
        save_dir: Directory to save model
        center: Center ID for saving model
    
    Returns:
        model: Trained model
        history: Dictionary containing training and validation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training model with {n_emb} embedding dimensions on {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Get only trainable parameters (FC, compressor, decompressor)
    trainable_params = model.get_trainable_params()
    
    # Print trainable parameter count
    total_params = sum(p.numel() for p in trainable_params)
    print(f"Training {total_params:,} parameters")
    
    # Set up optimizer (only for trainable parameters)
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-5)
    
    # Loss function from the provided code
    criterion = total_loss
    
    # Tracking metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_recon_loss': [],
        'val_recon_loss': [],
        'train_cls_acc': [],
        'val_cls_acc': []
    }
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model = None
    
    for epoch in range(num_epochs):
        # ========== Training ==========
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_cls_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels, _ in train_loader:  # Assuming dataloader returns (images, labels, paths)
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstructions, logits = model(images)
            
            # Calculate loss
            loss = criterion(reconstructions, images, logits, labels)
            
            # Also calculate individual components for tracking
            recon_loss = F.mse_loss(reconstructions, images)
            cls_loss = F.cross_entropy(logits, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_cls_loss += cls_loss.item()
            
            # Calculate accuracy
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Calculate average metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_recon_loss = train_recon_loss / len(train_loader)
        train_accuracy = 100. * correct / total if total > 0 else 0
        
        history['train_loss'].append(avg_train_loss)
        history['train_recon_loss'].append(avg_train_recon_loss)
        history['train_cls_acc'].append(train_accuracy)
        
        # ========== Validation ==========
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_cls_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                reconstructions, logits = model(images)
                
                # Calculate loss
                loss = criterion(reconstructions, images, logits, labels)
                
                # Also calculate individual components for tracking
                recon_loss = F.mse_loss(reconstructions, images)
                cls_loss = F.cross_entropy(logits, labels)
                
                # Track metrics
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_cls_loss += cls_loss.item()
                
                # Calculate accuracy
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Calculate average metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        val_accuracy = 100. * correct / total if total > 0 else 0
        
        history['val_loss'].append(avg_val_loss)
        history['val_recon_loss'].append(avg_val_recon_loss)
        history['val_cls_acc'].append(val_accuracy)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Train Recon: {avg_train_recon_loss:.4f} | '
              f'Train Acc: {train_accuracy:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f} | '
              f'Val Recon: {avg_val_recon_loss:.4f} | '
              f'Val Acc: {val_accuracy:.2f}%')
        
        # Check if this is the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)
            early_stop_counter = 0
            
            # Save the best model if a save directory is specified
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                model_name = f'vgg_autoencoder_emb{n_emb}'
                if center is not None:
                    model_name += f'_center{center}'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, os.path.join(save_dir, f'{model_name}.pth'))
                print(f"Saved best model to {os.path.join(save_dir, model_name)}.pth")
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{patience}")
            
        # Early stopping
        if early_stop_counter >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    # Use the best model
    if best_model is not None:
        return best_model, history
    
    return model, history


def extract_embeddings(model, data_loader, center, device, save_dir):
    """Extract and save embeddings using the trained model"""
    model.eval()
    model = model.to(device)
    
    names_list = []
    labels_list = []
    all_embeddings = []
    
    print(f"Extracting embeddings for center {center}...")
    
    with torch.no_grad():
        for images, labels, paths in data_loader:
            images = images.to(device)
            
            # Extract embeddings
            embeddings = model(images, embedding=True)
            
            # Get image names
            image_names = [p.split('/')[-1].split('.')[0] for p in paths]
            
            # Store results
            names_list.extend(image_names)
            labels_list.append(labels)
            all_embeddings.append(embeddings.cpu())
    
    # Concatenate all embeddings and labels
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(labels_list, dim=0).numpy()
    
    # Create embeddings directory if it doesn't exist
    os.makedirs(os.path.join(save_dir, "embeddings"), exist_ok=True)
    
    # Save embeddings individually
    for i, name in enumerate(names_list):
        emb_path = os.path.join(save_dir, "embeddings", f'center_{center}_{name}')
        np.save(emb_path, all_embeddings[i])
    
    # Save labels
    labels_df = pd.DataFrame({
        "Name": names_list,
        "Label": all_labels
    })
    labels_df.to_csv(os.path.join(save_dir, f'center_{center}_labels.csv'), index=False)
    print(f"Saved {len(names_list)} embeddings for center {center}")


def visualize_results(history, n_emb, center, save_dir):
    """Visualize and save training metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss curves
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Total Loss (Emb={n_emb}, Center={center})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(history['train_recon_loss'], label='Train Recon Loss')
    plt.plot(history['val_recon_loss'], label='Val Recon Loss')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'loss_curves_center{center}_emb{n_emb}.png'))
    
    # Accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_cls_acc'], label='Train Accuracy')
    plt.plot(history['val_cls_acc'], label='Val Accuracy')
    plt.title(f'Classification Accuracy (Emb={n_emb}, Center={center})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'accuracy_center{center}_emb{n_emb}.png'))
    
    plt.close('all')


def main(center, n_emb=256, data_path=None, save_dir=None):
    """Main training function for a specific center and embedding size"""
    if data_path is None:
        data_path = './data'
    if save_dir is None:
        save_dir = os.path.join(data_path, 'models')
    
    print(f"Training VGG Autoencoder for center {center} with {n_emb} embedding dimensions")
    
    # Initialize model
    configs = get_configs('vgg16')
    model = VGGAutoEncoder(configs, n_emb)
    
    # Load data
    train_data = ISICdataset.FedIsic2019(center=center, train=True, pooled=False, data_path=data_path)
    val_data = ISICdataset.FedIsic2019(center=center, train=False, pooled=False, data_path=data_path)
    
    train_loader = dl(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = dl(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train model
    trained_model, history = train_vgg_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_emb=n_emb,
        device=device,
        num_epochs=50,
        learning_rate=0.001,
        patience=10,
        save_dir=save_dir,
        center=center
    )
    
    # Visualize results
    visualize_results(history, n_emb, center, save_dir)
    
    # Extract and save embeddings
    extract_embeddings(trained_model, train_loader, center, device, save_dir)
    extract_embeddings(trained_model, val_loader, center, device, save_dir)
    
    return history



if __name__ == '__main__':
    BATCH_SIZE = 32
    DATA_PATH = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl/data/ISIC'
    SAVE_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl/data/ISIC'
    centers = range(6)
    for center in centers:
        main(center, n_emb=256, data_path=DATA_PATH, save_dir=SAVE_DIR)
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import json
import os
from torch.utils.data import DataLoader, TensorDataset
from emnist import extract_training_samples
import torch.nn.functional as F

# Paths & Hyperparameters
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl'
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes =62

# Load EMNIST dataset
images, labels = extract_training_samples('byclass')
images = images[:5000]
labels = labels[:5000]
images = images.astype(np.float32) / 255.0  # Normalize to [0,1]
images = np.expand_dims(images, axis=1)  # Add channel dimension (1,28,28)

# Convert to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(images), torch.tensor(labels))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, n_emb, num_classes):
        super(Autoencoder, self).__init__()
        self.n_emb = n_emb
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (14x14)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (7x7)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Linear(64 * 7 * 7, self.n_emb),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.n_emb, self.num_classes),
            nn.Softmax(dim=1))
        

        self.expand = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.n_emb, 64 * 7 * 7),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # (7x7 -> 14x14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # (14x14 -> 28x28)
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),  # Keep 28x28
            nn.Sigmoid()
        )

    def forward(self, x, return_logits=False):
        # Encoding
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        embeddings = self.bottleneck(x_flat)
        
        # Decoding
        x = self.expand(embeddings)
        x = x.view(x.size(0), 64, 7, 7)
        reconstruction = self.decoder(x)
        
        # Classification
        logits = self.classifier(embeddings)
        
        if return_logits:
            return reconstruction, logits
        return reconstruction

    def get_embedding(self, x):
        with torch.no_grad():
            return self.bottleneck(self.encoder(x).view(x.size(0), -1))
        
def total_loss(recon_x, x, logits, labels, alpha=0.5):
    recon_loss = F.mse_loss(recon_x, x)
    cls_loss = F.cross_entropy(logits, labels)
    return recon_loss + alpha * cls_loss

class EMNISTAutoencoder(nn.Module):
    def __init__(self, n_emb=64, num_classes=62): 
        super(EMNISTAutoencoder, self).__init__()
        self.n_emb = n_emb
        self.num_classes = num_classes
        
        # Encoder layers split into blocks for skip connections
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )  # 28x28x16
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 14x14x32
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 7x7x64
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, self.n_emb),
            nn.BatchNorm1d(self.n_emb)
        )
        
        # Classifier
        self.classifier = nn.Linear(self.n_emb, self.num_classes)
        
        # Decoder
        self.expand = nn.Sequential(
            nn.Linear(self.n_emb, 64 * 7 * 7),
            nn.ReLU()
        )
        
        # Decoder block 1: 7x7 -> 14x14
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 14x14
        )
        
        # Decoder block 2: 14x14 -> 28x28
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),  # +32 for skip connection
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 28x28
        )
        
        # Final reconstruction
        self.dec_final = nn.Sequential(
            nn.Conv2d(32 + 16, 8, 3, padding=1),  # +16 for skip connection
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, return_logits=False):
        # Encoding
        e1 = self.enc1(x)      # 28x28x16
        e2 = self.enc2(e1)     # 14x14x32
        e3 = self.enc3(e2)     # 7x7x64
        
        # Bottleneck
        x_flat = e3.view(e3.size(0), -1)
        embeddings = self.bottleneck(x_flat)
        embeddings = F.normalize(embeddings, p=2, dim=1) 
        # Classification
        logits = self.classifier(embeddings)
        
        # Decoding with skip connections
        d = self.expand(embeddings)
        d = d.view(d.size(0), 64, 7, 7)
        
        d = self.dec1(d)  # 14x14x64
        d = torch.cat([d, e2], dim=1)  # Skip connection from enc2
        
        d = self.dec2(d)  # 28x28x32
        d = torch.cat([d, e1], dim=1)  # Skip connection from enc1
        
        reconstruction = self.dec_final(d)  # 28x28x1
        
        if return_logits:
            return reconstruction, logits
        return reconstruction

    def get_embedding(self, x):
        with torch.no_grad():
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            x_flat = e3.view(e3.size(0), -1)
            embeddings = self.bottleneck(x_flat)
            return  F.normalize(embeddings, p=2, dim=1)

def total_loss(recon_x, x, logits, labels, alpha=0.6, beta=0.2, gamma=0.2):
    # Pixel-wise reconstruction loss
    mse_loss = F.mse_loss(recon_x, x)
    
    # Edge preservation loss for handwritten characters
    # Sobel filters to detect edges
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32).reshape(1, 1, 3, 3).to(x.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32).reshape(1, 1, 3, 3).to(x.device)
    
    # Pad inputs for convolution
    x_padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
    recon_padded = F.pad(recon_x, (1, 1, 1, 1), mode='replicate')
    
    # Calculate gradients using Sobel
    orig_grad_x = F.conv2d(x_padded, sobel_x)
    orig_grad_y = F.conv2d(x_padded, sobel_y)
    recon_grad_x = F.conv2d(recon_padded, sobel_x)
    recon_grad_y = F.conv2d(recon_padded, sobel_y)
    
    # Edge loss (L1 to focus on structure)
    edge_loss = (F.l1_loss(recon_grad_x, orig_grad_x) + 
                 F.l1_loss(recon_grad_y, orig_grad_y)) / 2.0
    
    # Classification loss
    cls_loss = F.cross_entropy(logits, labels)
    
    # Total loss
    total = alpha * mse_loss + beta * edge_loss + gamma * cls_loss
    
    return total

# Train autoencoder
def train_autoencoder(n_emb):
    model = Autoencoder(n_emb, num_classes).to(DEVICE)
    checkpoint_path = f'{ROOT_DIR}/data/EMNIST/model_checkpoint_{n_emb}.pth'
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))

    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_val_loss = np.inf
    patience = 10
    no_improvement_count = 0

    train_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0
        num_batches = 0
        
        for inputs, label in train_loader:
            inputs = inputs.to(DEVICE)
            label = label.to(DEVICE)
            recon, logits = model(inputs, return_logits=True)
            loss = total_loss(recon, inputs, logits, label)
            train_loss_sum += loss.item()
            num_batches += 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss_sum / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}")

        # Early stopping
        if avg_train_loss < best_val_loss:
            torch.save(model.state_dict(), checkpoint_path)
            best_val_loss = avg_train_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Stopping early after {patience} epochs without improvement.")
                break

    return best_val_loss, train_losses

# Main execution
def main():
    print(f"Using device: {DEVICE}")
    n_embs = [10,20,50,100,250]
    losses = {n_emb: train_autoencoder(n_emb) for n_emb in n_embs}

    file_path = f'{ROOT_DIR}/data/EMNIST/losses.json'

    # Load existing data if file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                existing_losses = json.load(f)
            except json.JSONDecodeError:
                existing_losses = {} 
    else:
        existing_losses = {}

    # Update losses dictionary
    existing_losses.update(losses)

    # Save updated data
    with open(file_path, 'w') as f:
        json.dump(existing_losses, f) 

if __name__ == '__main__':
    main()

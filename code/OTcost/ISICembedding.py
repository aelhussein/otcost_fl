global ROOT_DIR
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl'
global DATA_DIR
DATA_DIR = f'{ROOT_DIR}/data/ISIC'
import math

import sys
import json
import os
import torch
sys.path.append(f'{ROOT_DIR}/code/OTcost')
import torch.nn.functional as F
import torch.nn as nn
import ISICdataset
from torch.utils.data import DataLoader as dl
from torch.optim.lr_scheduler import ExponentialLR
import copy
from multiprocessing import Pool
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import copy
import time
import gc
import numpy as np
import os
from sklearn.metrics import balanced_accuracy_score

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):
    """Residual block for the encoder and decoder"""
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.downsample = None
        
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        
        out = avg_out + max_out
        return torch.sigmoid(out).view(x.size(0), x.size(1), 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        # Generate average and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along the channel dimension
        x = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid
        x = self.conv(x)
        return torch.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_att(x)
        # Apply spatial attention
        x = x * self.spatial_att(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, n_emb=128, input_size=(200, 200)):
        super(Autoencoder, self).__init__()
        self.n_emb = n_emb
        self.n_classes = 8
        # Calculate bottleneck dimensions based on input size
        h, w = input_size
        h1, w1 = h // 2, w // 2         # After first pooling
        h2, w2 = h1 // 2, w1 // 2       # After second pooling
        h3, w3 = h2 // 2, w2 // 2       # After third pooling
        
        self.bottleneck_dim = (64, h3, w3)  # Increased channel depth for richer representation
        
        # Encoder with residual blocks and attention
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            ResBlock(32, 32)
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResBlock(32, 64),
            CBAM(64)  # Attention after downsampling
        )
        
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResBlock(64, 128),
            CBAM(128)
        )
        
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResBlock(128, 64),
            CBAM(64)
        )
        
        # Bottleneck with additional non-linearity
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.bottleneck_dim[0] * self.bottleneck_dim[1] * self.bottleneck_dim[2], 
                      self.n_emb * 2),  # Wider bottleneck
            nn.LeakyReLU(0.1),
            nn.Linear(self.n_emb * 2, self.n_emb),
            nn.LeakyReLU(0.1)
        )


        self.classifier = nn.Sequential(
            nn.Linear(self.n_emb, self.n_classes),
            nn.Softmax(dim=1))
        
        
        # Decoder with residual blocks and skip connections
        self.expand = nn.Sequential(
            nn.Linear(self.n_emb, self.n_emb * 2),
            nn.LeakyReLU(0.1),
            nn.Linear(self.n_emb * 2, self.bottleneck_dim[0] * self.bottleneck_dim[1] * self.bottleneck_dim[2]),
            nn.LeakyReLU(0.1)
        )
        
        self.dec1 = nn.Sequential(
            ResBlock(64, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        
        self.dec2 = nn.Sequential(
            ResBlock(128, 64), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        
        self.dec3 = nn.Sequential(
            ResBlock(64, 32), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32)
        )
        
        self.dec4 = nn.Sequential(
            ResBlock(32, 16),  
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
    def _normalize_embedding(self, embedding, normalize=True):
        """Optionally normalize embedding to unit hypersphere"""
        if normalize:
            return F.normalize(embedding, p=2, dim=1)
        return embedding
    
    def forward(self, x, get_embedding=False, normalize_embedding=True):
        # Encoder
        e1 = self.enc1(x)           # 32 x h x w
        e2 = self.enc2(e1)          # 64 x h/2 x w/2
        e3 = self.enc3(e2)          # 128 x h/4 x w/4
        e4 = self.enc4(e3)          # 64 x h/8 x w/8
        
        # Bottleneck
        x_flat = e4.view(e4.size(0), -1)
        embedding = self.bottleneck(x_flat)
        
        # Normalize embedding if requested
        embedding = self._normalize_embedding(embedding, normalize_embedding)
        logits = self.classifier(embedding)

        if get_embedding:
            return embedding
        
        # Decoder
        x = self.expand(embedding)
        x = x.view(x.size(0), *self.bottleneck_dim)
        
        x = self.dec1(x)                    # 128 x h/4 x w/4
        #x = torch.cat([x, e3], dim=1)       # Skip connection from enc3
        
        x = self.dec2(x)                    # 64 x h/2 x w/2
        #x = torch.cat([x, e2], dim=1)       # Skip connection from enc2
        
        x = self.dec3(x)                    # 32 x h x w
        #x = torch.cat([x, e1], dim=1)       # Skip connection from enc1
        
        x = self.dec4(x)                    # 3 x h x w
        
        return x, logits, embedding


def contrastive_embedding_loss(embeddings, labels, margin=1.0):
    """Encourage same-class embeddings to be close, different ones far apart"""
    batch_size = embeddings.size(0)
    if batch_size <= 1:
        return torch.tensor(0.0, device=embeddings.device)
        
    # Calculate pairwise distances
    dist_matrix = torch.cdist(embeddings, embeddings)
    
    # Create mask for positive and negative pairs
    labels_matrix = labels.view(-1, 1) == labels.view(1, -1)
    
    # Losses for positive and negative pairs
    pos_mask = labels_matrix.float() - torch.eye(batch_size, device=labels.device)
    pos_mask = torch.clamp(pos_mask, min=0)  # Make sure diagonal is 0
    
    neg_mask = 1.0 - labels_matrix.float()
    
    # Contrastive loss
    positive_loss = (pos_mask * dist_matrix).sum() / max(pos_mask.sum(), 1)
    negative_loss = (neg_mask * F.relu(margin - dist_matrix)).sum() / max(neg_mask.sum(), 1)
    
    return positive_loss + negative_loss


def skin_lesion_loss(embedding, recon_x, x, logits, label, alpha=0.7, beta=0.3, gamma=0.1, delta = 0.5):
    cls_loss = F.cross_entropy(logits,label, weight = torch.tensor([0.02004131, 0.00740534, 0.02524011, 0.0967392 , 0.0345725 , 0.35093259, 0.33151339, 0.13355555], dtype=torch.float32).to(x.device)) 
    con_loss = contrastive_embedding_loss(embedding, label, margin=1.0)
    # Basic reconstruction loss (pixel-wise)
    mse_loss = F.mse_loss(recon_x, x)
    
    # Edge preservation loss (gradient-based)
    # Calculate gradients in x and y directions
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
    
    # Apply to each channel
    edge_loss = 0
    for c in range(3):  # RGB channels
        x_c = x[:, c:c+1]
        recon_x_c = recon_x[:, c:c+1]
        
        # Pad for convolution
        x_c_pad = F.pad(x_c, (1, 1, 1, 1), mode='replicate')
        recon_x_c_pad = F.pad(recon_x_c, (1, 1, 1, 1), mode='replicate')
        
        # Compute gradients
        x_grad_x = F.conv2d(x_c_pad, sobel_x)
        x_grad_y = F.conv2d(x_c_pad, sobel_y)
        recon_grad_x = F.conv2d(recon_x_c_pad, sobel_x)
        recon_grad_y = F.conv2d(recon_x_c_pad, sobel_y)
        
        # Compute loss on gradients
        edge_loss += F.l1_loss(x_grad_x, recon_grad_x) + F.l1_loss(x_grad_y, recon_grad_y)
    
    edge_loss /= 3.0  # Average across channels
    
    x_f32 = x.to(torch.float32)
    recon_x_f32 = recon_x.to(torch.float32)
    
    # Calculate frequency loss in float32
    fft_real = torch.fft.rfft2(x_f32, norm='ortho')
    fft_recon = torch.fft.rfft2(recon_x_f32, norm='ortho')
    
    freq_loss = F.l1_loss(fft_recon.real, fft_real.real) + \
                F.l1_loss(fft_recon.imag, fft_real.imag)
    
    # Convert back to original dtype
    freq_loss = freq_loss.to(x.dtype)
    
    # Combine losses
    total_loss =  mse_loss #alpha * mse_loss + beta * edge_loss + gamma * freq_loss #+ delta * con_loss
    if torch.isnan(total_loss):
        print(f"WARNING: NaN detected in loss calculation!")
        print(f"Loss components - MSE: {mse_loss}, Edge: {edge_loss}, Freq: {freq_loss}, Content: {con_loss}")
        print(embedding)

    
    return total_loss


def calculate_balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

def train_autoencoder(n_emb, model_class=None, data_dir=None, 
                      device=None, batch_size=BATCH_SIZE, num_workers=4,
                      max_epochs=50, patience=10):
    # Default values and setup
    if model_class is None:
        model_class = Autoencoder
    if data_dir is None:
        data_dir = DATA_DIR
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print configuration
    print(f"Training autoencoder with {n_emb} embedding dimensions")
    print(f"Device: {device}, Batch size: {batch_size}, Workers: {num_workers}")
    
    # Initialize model
    model = model_class(n_emb)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    checkpoint_path = f'{DATA_DIR}/model_checkpoint_{n_emb}.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")

    
    # Cosine annealing scheduler (better than exponential for convergence)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Load data efficiently - use pin_memory for faster host to GPU copies
    train_data = ISICdataset.FedIsic2019(train=True, pooled=True, data_path=data_dir)
    val_data = ISICdataset.FedIsic2019(train=False, pooled=True, data_path=data_dir)
    
    # Create dataloaders with efficient settings
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size * 2,  # Larger batches for validation (no gradients needed)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Early stopping parameters
    early_stopping_counter = 0
    best_val_loss = float('inf')
    
    # Tracking metrics
    train_losses = []
    val_losses = []
    train_balanced_acc = []
    val_balanced_acc = []
    
    # Get an appropriate validation batch count (around 10-20% of dataset)
    total_val_batches = len(val_loader)
    val_subset_size = max(8, min(int(total_val_batches * 0.2), 25))
    
    # Memory tracking
    peak_memory = 0
    
    # Training loop
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        batch_times = []
        
        # For balanced accuracy calculation
        all_train_preds = []
        all_train_labels = []
        
        # Reset GPU cache at start of epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        epoch_start = time.time()
        for i, (image, label, _) in enumerate(train_loader):
            batch_start = time.time()
            
            # Pre-process on CPU (transpose operation)
            image = image.transpose(2, 1)
            
            # Transfer to device
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)  # More memory efficient than just zero
            
            # Forward pass with automatic mixed precision
            with autocast(enabled=device.type=='cuda'):
                reconstructed, logits, embedding = model(image)
                loss = skin_lesion_loss(embedding, reconstructed, image, logits, label)
            
            # Backward and optimize with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Track metrics
            train_loss += loss.item()
            
            # Calculate predictions for balanced accuracy
            _, predicted = logits.max(1)
            
            # Store predictions and labels for balanced accuracy calculation
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(label.cpu().numpy())
            
            # Track batch time
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Track GPU memory usage
            if device.type == 'cuda':
                current_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                peak_memory = max(peak_memory, current_memory)
            
            # Print progress 
            if (i + 1) % 1 == 0:
                print(f"Epoch {epoch}, Batch {i+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Batch time: {batch_time:.3f}s")
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Calculate training balanced accuracy
        train_bal_acc = calculate_balanced_accuracy(all_train_labels, all_train_preds)
        train_balanced_acc.append(train_bal_acc)
        
        # Update learning rate once per epoch
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        # For balanced accuracy calculation
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for i, (image, label, _) in enumerate(val_loader):
                if i >= val_subset_size:
                    break
                    
                # Pre-process on CPU
                image = image.transpose(2, 1)
                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                
                # Forward pass
                reconstructed, logits, embedding = model(image)
                loss = skin_lesion_loss(embedding, reconstructed, image, logits, label)
                
                # Track validation loss
                val_loss += loss.item()
                
                # Calculate predictions for balanced accuracy
                _, predicted = logits.max(1)
                
                # Store predictions and labels for balanced accuracy calculation
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(label.cpu().numpy())
        
        # Calculate average validation loss
        avg_val_loss = val_loss / min(val_subset_size, len(val_loader))
        val_losses.append(avg_val_loss)
        
        # Calculate validation balanced accuracy
        val_bal_acc = calculate_balanced_accuracy(all_val_labels, all_val_preds)
        val_balanced_acc.append(val_bal_acc)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        avg_batch_time = np.mean(batch_times)
        
        # Print epoch summary with balanced accuracy
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s, Avg batch: {avg_batch_time:.3f}s")
        print(f"Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
        print(f"Train Balanced Acc: {train_bal_acc:.4f}, Val Balanced Acc: {val_bal_acc:.4f}")
        print(f"Peak GPU memory: {peak_memory:.2f} GB, Current LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint only if validation improves (without copying the whole model)
        if avg_val_loss < best_val_loss:
            print(f"Validation improved from {best_val_loss:.6f} to {avg_val_loss:.6f}")
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            
            # Save model state_dict only (more efficient than copying the model)
            checkpoint_path = f'{data_dir}/model_checkpoint_{n_emb}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                #'val_balanced_acc': val_bal_acc
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        else:
            early_stopping_counter += 1
            print(f"Validation did not improve. Early stopping counter: {early_stopping_counter}/{patience}")
            
        # Early stopping check
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        # Force garbage collection between epochs
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Load best model
    checkpoint = torch.load(f'{data_dir}/model_checkpoint_{n_emb}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return train_losses, val_losses


def main(n_emb, **kwargs):
    """Main function to train autoencoder with specified embedding size"""
    print(f"Starting training for n_emb={n_emb}")
    results = train_autoencoder(n_emb, **kwargs)
    print(f"Completed training for n_emb={n_emb}")
    return n_emb, results


if __name__ == '__main__':
    # Configuration
    n_embs = [1000]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For CPU training (parallel)
    if device.type == 'cpu':
        from multiprocessing import Pool
        cpu_count = int(os.environ.get('SLURM_CPUS_PER_TASK', 5))
        with Pool(cpu_count) as pool:
            results = pool.map(main, n_embs)
    # For GPU training (sequential)
    else:
        # Configure for maximum power efficiency
        torch.backends.cudnn.benchmark = True  # Improves speed for fixed-size inputs
        
        # Start with smaller embedding sizes (less memory intensive)
        n_embs.sort()
        results = []
        for n_emb in n_embs:
            results.append(main(n_emb))
    
    # Save all results
    losses = {}
    for n_emb, loss in results:
        losses[str(n_emb)] = {
            'train_losses': loss[0],
            'val_losses': loss[1]
        }
    
    import json
    with open(f'{ROOT_DIR}/data/ISIC/losses_2.json', 'w') as f:
        json.dump(losses, f)
    print("Losses saved to file.")

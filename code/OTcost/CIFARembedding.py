import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl'
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Download data
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR100(root=f'{ROOT_DIR}/data/CIFAR/', train=True, download=True, transform=transform)

all_images = train_dataset.data  # Shape: (50000, 32, 32, 3)
all_labels = np.array(train_dataset.targets)  # Convert to numpy

# Dictionary to store selected indices
selected_indices = []
num_samples_per_class = 100  # Limit to 100 samples per class
num_classes = 100

# Select 100 samples per class
for class_id in range(num_classes):
    class_indices = np.where(all_labels == class_id)[0]  # Get indices of this class
    chosen_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)  # Randomly select 100
    selected_indices.extend(chosen_indices)

# Create new dataset with only selected samples
subset_images = all_images[selected_indices]
subset_labels = all_labels[selected_indices]

# Convert to Tensor dataset
subset_dataset = torch.utils.data.TensorDataset(torch.tensor(subset_images).permute(0, 3, 1, 2).float() / 255.0,  # Normalize
                                                torch.tensor(subset_labels))

# Load into DataLoader
BATCH_SIZE = 64
train_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=True)

#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


class Autoencoder(nn.Module):
    def __init__(self, n_emb=128, num_classes=100):
        super(Autoencoder, self).__init__()
        self.n_emb = n_emb
        self.num_classes = num_classes
        
        # Encoder (slightly simplified)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, self.n_emb),
            nn.BatchNorm1d(self.n_emb)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
                                        nn.Linear(self.n_emb, self.num_classes),
                                        nn.Softmax(dim=1))
        
        # Decoder
        self.expand = nn.Sequential(
            nn.Linear(self.n_emb, 64 * 8 * 8),
            nn.ReLU()
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 8x8 -> 16x16
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, 3, padding=1),  # +64 for skip connection
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 16x16 -> 32x32
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 32, 16, 3, padding=1),  # +32 for skip connection
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, get_embedding=False):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Bottleneck
        embedding = self.bottleneck(e3)
        embedding = F.normalize(embedding, p=2, dim=1) 
        
        if get_embedding:
            return embedding
            
        # Classification
        logits = self.classifier(embedding)
        
        # Decoder with skip connections
        d = self.expand(embedding)
        d = d.view(d.size(0), 64, 8, 8)
        
        d = self.dec3(d)
        d = torch.cat([d, e2], dim=1)  # Skip connection from encoder
        
        d = self.dec2(d)
        d = torch.cat([d, e1], dim=1)  # Skip connection from encoder
        
        reconstruction = self.dec1(d)
        
        return reconstruction, logits
        
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

#Train autoencoder
def train_autoencoder(n_emb):
    model = Autoencoder(n_emb, 100)
    if f'model_checkpoint_{n_emb}.pth' in os.listdir(f'{ROOT_DIR}/data/CIFAR/'):
        state_dict = torch.load(f'{ROOT_DIR}/data/CIFAR/model_checkpoint_{n_emb}.pth')
        model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_train_loss = np.inf
    patience = 100
    no_improvement_count = 0 

    train_losses = []
    for epoch in range(EPOCHS):
        train_loss_sum = 0
        num_batches = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs, logits = model(inputs)
            train_loss = total_loss(outputs, inputs, logits, labels)
            train_loss_sum += train_loss.item()
            num_batches += 1
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        avg_train_loss = train_loss_sum / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)

        # Early stopping
        if avg_train_loss < best_train_loss:
            model.to('cpu')
            torch.save(model.state_dict(), f'{ROOT_DIR}/data/CIFAR/model_checkpoint_{n_emb}.pth')
            model.to(DEVICE)
            best_train_loss = avg_train_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Stopping early after {patience} epochs without improvement.")
                break
    return best_train_loss, train_losses

def main():
    print(DEVICE)
    n_embs = [10,50,100,250]
    losses = {n_emb: train_autoencoder(n_emb) for n_emb in n_embs}

    file_path = f'{ROOT_DIR}/data/CIFAR/losses.json'

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
    
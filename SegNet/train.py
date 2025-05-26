import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from model import SegNet
from vgg16_model import SegNetVGG16

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define dataset class
from PIL import Image

class WaterSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get all image files
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Get image path
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        
        # Get mask path (replace extension and add 'm' suffix)
        img_name = self.img_files[idx].split('.')[0]
        mask_name = img_name.replace('image', 'mask') + 'm.png'
        if 'class' in img_name:
          mask_name = img_name + 'm.png'
        else:
          mask_name = img_name.replace('image', 'mask') + 'm.png'

        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure image is resized to 512x512
        image = cv2.resize(image, (512, 512))
        
        # Convert numpy image to PIL Image for torchvision transforms compatibility
        image = Image.fromarray(image)
        
        # Load mask and convert to binary (water=1, non-water=0)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512))
        mask = (mask > 127).astype(np.float32)  # Threshold to binary
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Default transformation
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])(image)
        
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # Add channel dimension
        
        return image, mask


# IoU (Intersection over Union) metric
def iou_score(output, target):
    smooth = 1e-6
    output = torch.sigmoid(output) > 0.5
    output = output.view(-1).float()
    target = target.view(-1).float()
    
    intersection = (output * target).sum()
    union = output.sum() + target.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            running_loss += loss.item()
            running_iou += iou_score(outputs, masks).item()
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_iou = running_iou / len(train_loader)
        train_losses.append(epoch_train_loss)
        train_ious.append(epoch_train_iou)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        running_val_iou = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                val_loss = criterion(outputs, masks)
                
                running_val_loss += val_loss.item()
                running_val_iou += iou_score(outputs, masks).item()
        
        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_iou = running_val_iou / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_ious.append(epoch_val_iou)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_train_loss:.4f}, Train IoU: {epoch_train_iou:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val IoU: {epoch_val_iou:.4f}")
    
    return train_losses, val_losses, train_ious, val_ious

# Plot training and validation metrics
def plot_metrics(train_losses, val_losses, train_ious, val_ious):
    # Plot loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Training')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_ious, label='Training IoU')
    plt.plot(val_ious, label='Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('IoU over Training')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

# Visualize predictions
def visualize_predictions(model, val_loader, num_samples=5):
    model.eval()
    plt.figure(figsize=(15, 5 * num_samples))
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            image = images[0].to(device)
            true_mask = masks[0].cpu().numpy().squeeze()
            
            output = model(image.unsqueeze(0))
            pred_mask = (output.cpu().numpy().squeeze() > 0.5).astype(np.uint8)
            
            # Denormalize image for visualization
            img_np = images[0].permute(1, 2, 0).cpu().numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            plt.subplot(num_samples, 3, i*3 + 1)
            plt.imshow(img_np)
            plt.title('Input Image')
            plt.axis('off')
            
            plt.subplot(num_samples, 3, i*3 + 2)
            plt.imshow(true_mask, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.imshow(pred_mask, cmap='gray')
            plt.title('Prediction')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

def main():
    # Define paths
    train_img_dir = '../Dataset/Train/imgs'
    train_mask_dir = '../Dataset/Train/masks'
    val_img_dir = '../Dataset/Valid/imgs'
    val_mask_dir = '../Dataset/Valid/masks'
    
    # Check if directories exist
    if not all(os.path.exists(p) for p in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]):
        print("Dataset directories not found. Please check the paths.")
        return
    
    # Define transformations for training dataset
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets with transformations for training only
    train_dataset = WaterSegmentationDataset(train_img_dir, train_mask_dir, transform=train_transform)
    val_dataset = WaterSegmentationDataset(val_img_dir, val_mask_dir, transform=train_transform)
    
    # Create dataloaders (batch_size=4 as specified)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = SegNetVGG16().to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_losses, val_losses, train_ious, val_ious = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=50
    )
    
    # Plot metrics
    plot_metrics(train_losses, val_losses, train_ious, val_ious)
    
    # Visualize predictions
    print("Generating prediction visualizations...")
    visualize_predictions(model, val_loader)
    
    # Save the model
    torch.save(model.state_dict(), 'water_segmentation_model.pth')
    print("Model saved successfully.")

if __name__ == "__main__":
    main()

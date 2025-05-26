import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.metrics import jaccard_score
from shufflenetv2_model import BiSeNet, WaterSegLoss

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define dataset paths
data_dir = './Dataset'
train_img_dir = os.path.join(data_dir, 'Train/imgs')
train_mask_dir = os.path.join(data_dir, 'Train/masks')
val_img_dir = os.path.join(data_dir, 'Valid/imgs')
val_mask_dir = os.path.join(data_dir, 'Valid/masks')
test_img_dir = os.path.join(data_dir, 'Test/imgs')
test_mask_dir = os.path.join(data_dir, 'Test/masks')

# Define transformations
image_size = 512  # This matches our ShuffleNetV2 model's expected input size

train_transform = A.Compose([
    A.Resize(image_size, image_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Custom Dataset class
class WaterSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get all image filenames
        self.img_filenames = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.img_filenames)
    
    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        
        # Get the corresponding mask filename
        base_name = os.path.splitext(img_filename)[0]
        
        # Handle different naming patterns
        if 'image' in base_name:
            # For format like "channel_image_014.jpg" -> "channel_mask_014m.png"
            base_name = base_name.replace('image', 'mask')
            mask_filename = f"{base_name}m.png"
        else:
            # For formats like "N02_6_0000030650.jpg" -> "N02_6_0000030650m.png"
            mask_filename = f"{base_name}m.png"
            
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Load mask and ensure it's binary (0 or 1)
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)  # Convert to binary mask (0 or 1)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # Add channel dimension
        
        return image, mask

# Create datasets and check if they exist
if not os.path.exists(train_img_dir) or not os.path.exists(train_mask_dir):
    raise FileNotFoundError(f"Training dataset not found at {train_img_dir} or {train_mask_dir}")

train_dataset = WaterSegmentationDataset(train_img_dir, train_mask_dir, transform=train_transform)
val_dataset = WaterSegmentationDataset(val_img_dir, val_mask_dir, transform=val_transform)
test_dataset = WaterSegmentationDataset(test_img_dir, test_mask_dir, transform=val_transform)

# Verify dataset sizes
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Create dataloaders
batch_size = 8  # Adjust based on your GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize model - using ShuffleNetV2 backbone
model = BiSeNet(num_classes=1, backbone_scale='1.0').to(device)
print("Initialized BiSeNet with ShuffleNetV2 backbone")

# Initialize optimizer with weight decay to prevent overfitting
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

# Define criterion
criterion = WaterSegLoss(alpha=0.7, beta=0.15, gamma=0.15)

# Training function
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(loader)

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    iou_scores = []
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            # Calculate IoU (Jaccard Index)
            predicted = torch.sigmoid(outputs[0]) > 0.5
            masks_cpu = masks.cpu().numpy().squeeze(1).astype(bool)
            predicted_cpu = predicted.cpu().numpy().squeeze(1).astype(bool)
            
            for i in range(masks.size(0)):
                iou = jaccard_score(masks_cpu[i].flatten(), predicted_cpu[i].flatten(), average='binary')
                iou_scores.append(iou)
    
    avg_iou = np.mean(iou_scores)
    return val_loss / len(loader), avg_iou

# Function to save checkpoints
def save_checkpoint(model, optimizer, epoch, filename):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")

# Training loop
num_epochs = 50
best_val_iou = 0.0

# Create a directory to save models
os.makedirs('checkpoints', exist_ok=True)

# Training history for plotting
train_losses = []
val_losses = []
val_ious = []

print(f"Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    train_losses.append(train_loss)
    
    # Validate
    val_loss, val_iou = validate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_ious.append(val_iou)
    
    # Update learning rate
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, LR: {current_lr:.6f}')
    
    # Save best model
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        save_checkpoint(model, optimizer, epoch, 'checkpoints/bisenet_shufflenetv2_best_model.pth')
        print(f'Saved best model with IoU: {best_val_iou:.4f}')
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        save_checkpoint(model, optimizer, epoch, f'checkpoints/bisenet_shufflenetv2_epoch_{epoch+1}.pth')

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(val_ious, label='Val IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.title('Validation IoU')

plt.tight_layout()
plt.savefig('training_history.png')
print("Training history saved to training_history.png")

# Load best model for evaluation
print("Loading best model for evaluation...")
checkpoint = torch.load('checkpoints/bisenet_shufflenetv2_best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate on test set
test_loss, test_iou = validate(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}')

# Visualization function
def visualize_predictions(model, dataset, device, num_samples=5):
    """
    Visualize model predictions compared to ground truth masks
    """
    model.eval()
    # Create a figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples*5))
    
    # Select random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get image and mask
            image, mask = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            
            # Get prediction
            outputs = model(image_tensor)
            prediction = torch.sigmoid(outputs[0]) > 0.5
            
            # Convert tensors to numpy arrays for visualization
            image_np = image.permute(1, 2, 0).cpu().numpy()
            # Denormalize image
            image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image_np = np.clip(image_np, 0, 1)
            
            mask_np = mask.squeeze().cpu().numpy()
            pred_np = prediction.squeeze().cpu().numpy()
            
            # Display
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title('Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_np, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('water_segmentation_results.png')
    print("Visualization saved to water_segmentation_results.png")

# Visualize some predictions on test set
try:
    print("Generating visualizations...")
    visualize_predictions(model, test_dataset, device)
except Exception as e:
    print(f"Error during visualization: {e}")
    print("Continuing with the rest of the script...")

# Inference function
def predict_mask(model, image_path, device):
    """
    Predict water segmentation mask for a single image
    
    Args:
        model: Trained BiSeNet model
        image_path: Path to the image
        device: Device to perform inference on (cuda/cpu)
        
    Returns:
        Original image, predicted binary mask
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = val_transform
    transformed = transform(image=np.array(image))
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        prediction = torch.sigmoid(outputs[0]) > 0.5
    
    # Convert to numpy
    prediction_np = prediction.squeeze().cpu().numpy().astype(np.uint8) * 255
    
    return image, prediction_np

# Example of how to use the inference function
# Uncomment to test on a specific image
# test_image_path = "path/to/test/image.jpg"
# if os.path.exists(test_image_path):
#     original_image, predicted_mask = predict_mask(model, test_image_path, device)
#     
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(original_image)
#     plt.title("Original Image")
#     plt.axis('off')
#     
#     plt.subplot(1, 2, 2)
#     plt.imshow(predicted_mask, cmap='gray')
#     plt.title("Predicted Mask")
#     plt.axis('off')
#     
#     plt.tight_layout()
#     plt.savefig('single_prediction.png')

# Save the model for future use
torch.save(model.state_dict(), 'water_segmentation_bisenet_shufflenetv2_final.pth')
print("Final model saved as water_segmentation_bisenet_shufflenetv2_final.pth")

print("Training and evaluation completed successfully!")
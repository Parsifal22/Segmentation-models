import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# Assuming your combined model is in DeepLab3+/model.py
from model import DeepLabV3Plus

# Placeholder Dataset Class (replace with your actual dataset)
class CustomSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.img_files.sort()
        self.mask_files.sort()
        assert len(self.img_files) == len(self.mask_files), "Number of images and masks must be the same"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Assuming grayscale mask

        # Apply transformations to image
        if self.transform:
            image = self.transform(image)

        # Apply resize and ToTensor to mask
        resize_transform = transforms.Resize((512, 512))
        to_tensor_transform = transforms.ToTensor()

        mask = resize_transform(mask)
        mask = to_tensor_transform(mask).squeeze(0).long() # Convert mask to tensor and ensure correct data type and shape

        return image, mask

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.squeeze(1).long()) # Adjust target shape if necessary
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def main():
    parser = argparse.ArgumentParser(description='Train DeepLabV3+ Segmentation Models')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory with training images')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory with training masks')
    parser.add_argument('--save_dir', type=str, default='DeepLab3+/models', help='Directory to save models')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transformations (adjust as needed)
    transform = transforms.Compose([
        transforms.Resize((512, 512)), # Resize to a consistent size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet standards
    ])

    # Load dataset
    train_dataset = CustomSegmentationDataset(img_dir=args.img_dir, mask_dir=args.mask_dir, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print(f"Dataset size: {len(train_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Drop last batch: True")

    # Define model configurations to train
    model_configs = [
        {'model_type': 'resnet', 'backbone': 'resnet50'},
        {'model_type': 'resnet', 'backbone': 'resnet101'},
        {'model_type': 'vgg'},
        {'model_type': 'xception'}
    ]

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    for config in model_configs:
        model_type = config['model_type']
        backbone = config.get('backbone', None) # Use .get for optional backbone

        print(f"Initializing and training model: {model_type}{' with backbone ' + backbone if backbone else ''}")

        # Initialize model
        model = DeepLabV3Plus(num_classes=args.num_classes)
        if backbone:
            model.construct(model_type=model_type, backbone=backbone)
            save_name = f"{model_type}_{backbone}_best_model.pth"
        else:
            model.construct(model_type=model_type)
            save_name = f"{model_type}_best_model.pth"

        model.to(device)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        best_loss = float('inf')
        save_path = os.path.join(args.save_dir, save_name)

        print(f"Starting training for {model_type}{' with backbone ' + backbone if backbone else ''}")

        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            train_loss = train(model, train_dataloader, criterion, optimizer, device)
            print(f"Train Loss: {train_loss:.4f}")

            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")

        print(f"Training finished for {model_type}{' with backbone ' + backbone if backbone else ''}")
        print("-" * 50) # Separator for clarity

    print("All model training finished.")

if __name__ == "__main__":
    main()

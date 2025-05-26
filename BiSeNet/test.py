import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import pandas as pd
from shufflenetv2_model import BiSeNet, WaterSegLoss

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone_scale = '1.0'  # ShuffleNetV2 scale

model_path = './shufflenetv2_bisenet.pth'
output_dir = './inf_results_bisenet_shufflenetv2'

test_img_dir = './test/imgs'
test_mask_dir = './test/masks'

# Function to overlay mask on image
def overlay_mask_on_image(image, mask, alpha=0.5, color=[0, 0, 255]):
    """
    Overlay a mask on an image with specified color and transparency
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    for i in range(3):
        mask_rgb[:, :, i] = mask * color[i]
    
    blended = (1 - alpha) * image + alpha * mask_rgb * 255
    return np.clip(blended, 0, 255).astype(np.uint8)

# Create metrics dataframe
metrics_data = {
    'Image': [],
    'IoU': [],
    'Precision': [],
    'Recall': [],
    'F1': [],
    'Inference_Time_ms': []
}

# Number of test images
total_images = len([f for f in os.listdir(test_img_dir) if f.endswith('.jpg')])
print(f"Testing on {total_images} images...")

os.makedirs(output_dir, exist_ok=True)

# Model setup
model = BiSeNet(num_classes=1, backbone_scale=backbone_scale)
model.to(device)

# Print model size (number of parameters)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
param_size_bytes = total_params * 4  # assuming float32 (4 bytes)
param_size_mb = param_size_bytes / (1024 ** 2)
print(f"Approximate model size: {param_size_mb:.2f} MB")

# Load model weights
try:
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    print(f"Successfully loaded model from {model_path}")
except Exception as e:
    print(f"Error loading model weights: {e}")
    print("Running with initialized weights!")

model.eval()

# Preprocessing (should match training transforms)
preprocess = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Process test images
for img_name in os.listdir(test_img_dir):
    if not img_name.endswith('.jpg'):
        continue
    
    img_path = os.path.join(test_img_dir, img_name)
    base_name = os.path.splitext(img_name)[0]
    mask_name = base_name + ".png"
    mask_path = os.path.join(test_mask_dir, mask_name)
    
    if not os.path.exists(mask_path):
        print(f"Mask {mask_path} not found. Skipping.")
        continue
    
    img = Image.open(img_path).convert('RGB')
    original_size = img.size
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        output, output_aux1, output_aux2 = model(img_tensor)
    inference_time = (time.time() - start_time) * 1000
    
    pred_mask = torch.sigmoid(output.squeeze()).cpu().numpy()
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    
    gt_mask = Image.open(mask_path).convert("L")
    gt_mask = gt_mask.resize((512, 512), Image.NEAREST)
    gt_mask_np = np.array(gt_mask) > 0
    
    y_true = gt_mask_np.flatten().astype(int)
    y_pred = (binary_mask.flatten() > 127).astype(int)
    
    iou = jaccard_score(y_true, y_pred, average='binary', zero_division=1)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=1)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=1)
    
    metrics_data['Image'].append(base_name)
    metrics_data['IoU'].append(iou)
    metrics_data['Precision'].append(precision)
    metrics_data['Recall'].append(recall)
    metrics_data['F1'].append(f1)
    metrics_data['Inference_Time_ms'].append(inference_time)
    
    img_resized = img.resize((512, 512))
    overlay_img = overlay_mask_on_image(img_resized, binary_mask, alpha=0.4, color=[0, 0, 255])
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    ax1.imshow(img_resized)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(np.array(gt_mask), cmap='gray')
    ax2.set_title('Ground Truth Mask')
    ax2.axis('off')
    
    ax3.imshow(pred_mask, cmap='gray')
    ax3.set_title('Predicted Mask (Probability)')
    ax3.axis('off')
    
    ax4.imshow(overlay_img)
    ax4.set_title(f'Overlay (IoU: {iou:.3f}, F1: {f1:.3f})')
    ax4.axis('off')
    
    plt.savefig(os.path.join(output_dir, f'comparison_{base_name}.png'))
    plt.close()

metrics_df = pd.DataFrame(metrics_data)
avg_metrics = metrics_df.mean(numeric_only=True)

avg_inference_time_ms = avg_metrics['Inference_Time_ms']
fps = 1000 / avg_inference_time_ms if avg_inference_time_ms > 0 else 0

with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
    f.write(f"Model Performance Metrics Summary\n")
    f.write(f"================================\n\n")
    f.write(f"Test Images: {total_images}\n")
    f.write(f"Average IoU: {avg_metrics['IoU']:.4f}\n")
    f.write(f"Average Precision: {avg_metrics['Precision']:.4f}\n")
    f.write(f"Average Recall: {avg_metrics['Recall']:.4f}\n")
    f.write(f"Average F1 Score: {avg_metrics['F1']:.4f}\n")
    f.write(f"Average Inference Time: {avg_metrics['Inference_Time_ms']:.2f} ms\n")
    f.write(f"Equivalent FPS: {fps:.2f}\n")
    f.write(f"\nNote: These metrics assume binary segmentation with threshold 0.5\n")

metrics_df.to_csv(os.path.join(output_dir, 'detailed_metrics.csv'), index=False)

print("\nMetrics Summary:")
print(f"Average IoU: {avg_metrics['IoU']:.4f}")
print(f"Average Precision: {avg_metrics['Precision']:.4f}")
print(f"Average Recall: {avg_metrics['Recall']:.4f}")
print(f"Average F1 Score: {avg_metrics['F1']:.4f}")
print(f"Average Inference Time: {avg_metrics['Inference_Time_ms']:.2f} ms")
print(f"Equivalent FPS: {fps:.2f} (compared to target 30 FPS)")
print(f"\nDetailed metrics saved to {os.path.join(output_dir, 'detailed_metrics.csv')}")
print(f"Testing complete. Results saved to {output_dir}")

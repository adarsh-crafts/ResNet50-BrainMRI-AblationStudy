import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

import os
import argparse
import json
import shutil
import copy
import time
import glob

parser = argparse.ArgumentParser(description='ResNet-50 Ablation Training')
parser.add_argument('--config_file', type=str, required=True,
                    help='Path to the JSON config file for the experiment.')
args = parser.parse_args()

# open JSON file
try:
    with open(args.config_file, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Config file not found at {args.config_file}")
    exit()

# create the directories
base_exp_dir = 'experiments'
exp_dir = os.path.join(base_exp_dir, config['run_name'])
weights_dir = os.path.join(exp_dir, 'weights')
checkpoint_dir = os.path.join(exp_dir, 'checkpoints')

os.makedirs(weights_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

shutil.copy(args.config_file, os.path.join(exp_dir, 'config.json'))

print(f'--- Starting Run: {config["run_name"]} ---')
print(f'Config loaded from {args.config_file}')
print(f'Model weights will be saved to {weights_dir}/best_model.pth')
print(f'Checkpoints will be saved to {checkpoint_dir}/')
print(f'Running with config: {json.dumps(config, indent=2)}')

# 1. setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 2. Dynamic transforms based on config
transforms_list = [
    transforms.Resize((240, 240)),
]

if config['augmentations']['rotation']:
    transforms_list.append(transforms.RandomRotation(10))
    print("Augmentation Enabled: RandomRotation")

if config['augmentations']['shift']:
    transforms_list.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.2)))
    print("Augmentation Enabled: RandomAffine (Height Shift)")

if config['augmentations']['flip']:
    transforms_list.append(transforms.RandomHorizontalFlip())
    print("Augmentation Enabled: RandomHorizontalFlip")

transforms_list.append(transforms.ToTensor())

train_val_transforms = transforms.Compose(transforms_list)

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 3. Data loaders
full_train_dataset = datasets.ImageFolder(
    root = config['train_dir'],
    transform = train_val_transforms
)

val_size = int(len(full_train_dataset) * 0.2)
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False
)

# load model
num_classes = len(full_train_dataset.class_to_idx)
print(f'Found {num_classes} classes.')

# Initialize model
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Handle pretrained weights or resume
start_epoch = 0
best_acc = 0.0

if config['model']['pretrained'] == 'resume':
    # Find latest checkpoint
    checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth')
    checkpoints = glob.glob(checkpoint_pattern)
    
    if checkpoints:
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.split('_epoch_')[1].split('.pth')[0]))
        latest_checkpoint = checkpoints[-1]
        
        print(f'Resuming from checkpoint: {latest_checkpoint}')
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        
        print(f'Resumed from epoch {checkpoint["epoch"]} with best acc: {best_acc:.4f}')
    else:
        print('No checkpoint found. Starting from scratch.')
        
elif config['model']['pretrained']:
    print('Loading ImageNet pretrained weights.')
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

model = model.to(device)

# define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['lr'])

# If resuming, load optimizer state
if config['model']['pretrained'] == 'resume' and checkpoints:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Optimizer state restored.')

# train and val
dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
best_model_wts = copy.deepcopy(model.state_dict())

print('Starting training...')

for epoch in range(start_epoch, config['epochs']): 
    print(f'Epoch {epoch+1}/{config["epochs"]}')
    print('-'*10)
    start_time = time.time()

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_fn(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    
    # Save checkpoint after each epoch
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'config': config
    }, checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')
    
    time_elapsed = time.time() - start_time
    print(f'Epoch completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print()

print(f'Training complete. Best val Acc: {best_acc:.4f}')
model.load_state_dict(best_model_wts)

# save
save_path = os.path.join(weights_dir, 'best_model.pth')
torch.save(model.state_dict(), save_path)
print(f'Best model weights saved to {save_path}')
print(f'--- Run {config["run_name"]} Finished ---')
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pipeline.common.config import Config
from tqdm import tqdm
import os

from pipeline.models.image_mol import ImageMol, get_support_model_names

def parse_args():
    parser = argparse.ArgumentParser(description="Train ImageMol Model")
    parser.add_argument("--cfg-path", type=str, default="path/to/drugchat.yaml", help="Path to config file.")
    parser.add_argument("--base-model", type=str, default="ResNet18", choices=get_support_model_names(), help="Base ResNet model.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to dataset.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for pretrained weights.")
    args = parser.parse_args()
    return args

def setup_dataloaders(data_dir, batch_size):
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader

def train_one_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(data_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss

def evaluate(model, criterion, data_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    epoch_loss = running_loss / len(data_loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Code running...")

    # Model setup
    model = ImageMol(baseModel=args.base_model, jigsaw_classes=101, label1_classes=2).to(device)
    if args.checkpoint:
        model.load_from_pretrained(args.checkpoint)

    # Data loaders
    train_loader, val_loader = setup_dataloaders(args.data_dir, args.batch_size)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training and validation loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Train
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
        print(f"Training Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    print("Training completed!")

if __name__ == "__main__":
    main()

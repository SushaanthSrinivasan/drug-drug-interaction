import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import json
import math
import logging

# from pipeline.models.image_mol import ImageMol, get_support_model_names

root_path = "." # local
# root_path = "/data" # k8

def get_support_model_names():
    return ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


def load_model(modelname="ResNet18", imageSize=224, num_classes=2):
    assert modelname in get_support_model_names()
    if modelname == "ResNet18":
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet34":
        model = torchvision.models.resnet34(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet50":
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet101":
        model = torchvision.models.resnet101(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet152":
        model = torchvision.models.resnet152(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception("{} is undefined".format(modelname))
    # emb_dim = model.fc.in_features
    return model


class ImageMol(nn.Module):
    def __init__(self, baseModel="ResNet18", jigsaw_classes=101, label1_classes=100, label2_classes=1000, label3_classes=10000):
        super(ImageMol, self).__init__()

        assert baseModel in get_support_model_names()

        self.baseModel = baseModel

        self.embedding_layer = nn.Sequential(*list(load_model(baseModel).children())[:-1])
        self.emb_dim = 512
        # self.bn = nn.BatchNorm1d(512)

        # self.jigsaw_classifier = nn.Linear(512, jigsaw_classes)
        # self.class_classifier1 = nn.Linear(512, label1_classes)
        # self.class_classifier2 = nn.Linear(512, label2_classes)
        # self.class_classifier3 = nn.Linear(512, label3_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.view(x.size(0), -1)
        return x

        # x1 = self.jigsaw_classifier(x)
        # x2 = self.class_classifier1(x)
        # x3 = self.class_classifier2(x)
        # x4 = self.class_classifier3(x)

        # return x, x1, x2, x3, x4

    def load_from_pretrained(self, url_or_filename):
        if os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Loading info: {}".format(msg))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

# if __name__ == "__main__":
#     ckpt = torch.load("ckpt/ImageMol.pth.tar", map_location="cpu")
#     net = ImageMol("ResNet18")
#     msg = net.load_state_dict(ckpt["state_dict"], strict=False)
#     print(msg)
#     x = torch.rand(2, 3, 224, 224)
#     y = net(x)
#     print(y.shape)


class DrugInteractionDataset(Dataset):
    def __init__(self, excel_file, smiles_to_path, transform=None):
        self.data = pd.read_excel(excel_file)
        self.smiles_to_path = smiles_to_path
        self.transform = transform
        self.interaction_map = {'low': 0, 'moderate': 1, 'high': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles1, smiles2 = row['Smiles_1'], row['Smiles_2']
        img1 = Image.open(self.smiles_to_path[smiles1]).convert('RGB')
        img2 = Image.open(self.smiles_to_path[smiles2]).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        interaction = self.interaction_map[row['interaction_type'].lower()]
        return img1, img2, torch.tensor(interaction)

class DrugInteractionClassifier(nn.Module):
    def __init__(self, base_model="ResNet18"):
        super(DrugInteractionClassifier, self).__init__()
        self.imagemol = ImageMol(baseModel=base_model)
        self.fc = nn.Linear(512 * 2, 3)  # 3 classes: low, moderate, high
        self.activation = nn.ReLU()

    def forward(self, x1, x2):
        emb1 = self.imagemol(x1)
        emb2 = self.imagemol(x2)
        concat_emb = torch.cat((emb1, emb2), dim=1)
        x = self.activation(concat_emb)
        x = self.fc(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description="Train Drug Interaction Classifier")
    parser.add_argument("--cfg-path", type=str, default="path/to/drugchat.yaml", help="Path to config file.")
    parser.add_argument("--base-model", type=str, default="ResNet18", choices=get_support_model_names(), help="Base ResNet model.")
    parser.add_argument("--excel-file", type=str, default=f"{root_path}/data/new_drug_drug_interaction_2.xlsx", help="Path to Excel file.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for pretrained weights.")
    return parser.parse_args()

def setup_dataloader(excel_file, smiles_to_path, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = DrugInteractionDataset(excel_file, smiles_to_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def train_one_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    running_loss = 0.0
    for img1, img2, labels in tqdm(data_loader, desc="Training"):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(img1, img2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * img1.size(0)
    return running_loss / len(data_loader.dataset)

def evaluate(model, criterion, data_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for img1, img2, labels in tqdm(data_loader, desc="Validation"):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * img1.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(data_loader.dataset), correct / total

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(os.getcwd())

    print("Code running...")

    smiles_to_path = {}
    with open(f"{root_path}/dataset/smiles_to_path.json", "r") as file:
        smiles_to_path = json.load(file)

    model = DrugInteractionClassifier(base_model=args.base_model).to(device)
    if args.checkpoint:
        model.imagemol.load_from_pretrained(args.checkpoint)

    for param in model.imagemol.parameters():
        param.requires_grad = False

    data_loader = setup_dataloader(args.excel_file, smiles_to_path, args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, criterion, optimizer, data_loader, device)
        val_loss, val_acc = evaluate(model, criterion, data_loader, device)
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    print("Training completed!")

if __name__ == "__main__":
    main()
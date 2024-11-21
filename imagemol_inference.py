import torch
import torchvision
import math
import os
import logging
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Draw
import json
from PIL import Image
import hashlib

hash_to_path = {}
with open("./dataset/hash_to_smi.json", "r") as f:
    hash_to_path = json.load(f)


def load_image(image_path):
    img = Image.open(image_path)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])
    img_tensor = transform(img)    
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

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

class DrugInteractionPredictor(nn.Module):
    def __init__(self, baseModel="ResNet18", num_classes=3):  # num_classes for high, moderate, low
        super(DrugInteractionPredictor, self).__init__()
        
        self.imagemol = ImageMol(baseModel)
        
        # Load pre-trained weights
        ckpt = torch.load("ckpt/ImageMol.pth.tar", map_location="cpu")
        self.imagemol.load_state_dict(ckpt["state_dict"], strict=False)
        
        # Freeze ImageMol parameters
        for param in self.imagemol.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(self.imagemol.emb_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x1, x2):
        # Get embeddings for both drugs
        emb1 = self.imagemol(x1)
        emb2 = self.imagemol(x2)
        
        # Concatenate embeddings
        combined_emb = torch.cat((emb1, emb2), dim=1)
        
        # Classify
        output = self.classifier(combined_emb)
        return output

def smiles_to_image(smiles, size=224):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(size, size))
    img_tensor = torchvision.transforms.ToTensor()(img)
    return img_tensor.unsqueeze(0)  # Add batch dimension

class ImageMol(nn.Module):
    def __init__(self, baseModel="ResNet18"):
        super(ImageMol, self).__init__()

        assert baseModel in get_support_model_names()

        self.baseModel = baseModel

        self.embedding_layer = nn.Sequential(*list(load_model(baseModel).children())[:-1])
        self.emb_dim = 512

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


if __name__ == "__main__":
    model = DrugInteractionPredictor()
    
    smiles1 = "CC(=O)OC1=CC=CC=C1C(=O)O"
    smiles2 = "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"
    
    smi1_hash = hashlib.md5(smiles1.encode()).hexdigest()
    smi2_hash = hashlib.md5(smiles2.encode()).hexdigest()

    img1 = load_image(hash_to_path(smi1_hash))
    img2 = load_image(hash_to_path(smi2_hash))
    
    with torch.no_grad():
        output = model(img1, img2)
        
    print(f"Interaction prediction: {output}")
    print(f"Predicted class: {torch.argmax(output)}")  # 0: low, 1: moderate, 2: high
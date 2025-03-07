import os
import json
import pickle
import torch

from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torch_geometric.data import Data, Batch


class MultimodalDataset(Dataset):
    def __init__(self, datapath, use_image=True, use_graph=False, image_size=224, is_train=False) -> None:
        super().__init__()
        self.use_image = use_image
        self.use_graph = use_graph
        jsonpath = os.path.join(datapath, "new_smiles_img_qa.json")
        print(f"Using {jsonpath=}")
        with open(jsonpath, "rt") as f:
            meta = json.load(f)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        if is_train:
            self.transforms = transforms.Compose([
                transforms.RandomRotation((0, 180), fill=255),
                transforms.RandomResizedCrop(image_size, (0.5, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.CenterCrop(image_size), 
                transforms.ToTensor(),
                normalize,
            ])
        if use_graph:
            with open(os.path.join(datapath, "graph_smi.pkl"), "rb") as f:
                graphs = pickle.load(f)
                print(f"Graphs loaded successfully with {len(graphs)} entries.")
        self.images = {}
        self.data = []
        self.graphs = {}
        for idx, rec in meta.items():
            if use_image:
                img_file_1 = 'img_{}_0.png'.format(idx)
                img_file_2 = 'img_{}_1.png'.format(idx)
                image_path_1 = os.path.join(datapath, img_file_1)
                image_path_2 = os.path.join(datapath, img_file_2)
                image1 = Image.open(image_path_1).convert("RGB")
                image2 = Image.open(image_path_2).convert("RGB")
                # img = self.transforms(image)
                self.images[idx] = [image1, image2]
            smi, qa = rec
            if use_graph:
                g1 = graphs[smi]["graph1"]
                g2 = graphs[smi]["graph2"]
                graph1 = Data(x=torch.asarray(g1['node_feat']), edge_index=torch.asarray(g1['edge_index']), edge_attr=torch.asarray(g1['edge_feat']))
                graph2 = Data(x=torch.asarray(g2['node_feat']), edge_index=torch.asarray(g2['edge_index']), edge_attr=torch.asarray(g2['edge_feat']))
                self.graphs[idx] = [graph1, graph2]
            qa = [(idx, qa_pair) for qa_pair in qa]
            self.data.extend(qa)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        idx, qa_pair = self.data[index]
        out = {"question": qa_pair[0], "text_input": str(qa_pair[1])}
        # out = {"question": None, "text_input": str(qa_pair[0])}
        if self.use_image:
            imgs = [self.transforms(img) for img in self.images[idx]]
            out.update({"img": imgs})
        if self.use_graph:
            out.update({"graph": self.graphs[idx]})
        return out
    
    @staticmethod
    def collater(samples):
        # if samples[0].get("question") is not None:
        #     qq = [x["question"] for x in samples]
        # else:
        #     qq = Non
        qq = [x["question"] for x in samples]
        aa = [x["text_input"] for x in samples]
        out = {"question": qq, "text_input": aa}
        # print(f"Out: {out}")
        # Handle images if they exist
        if "img" in samples[0]:
            imgs_collated = [default_collate(sample["img"]) for sample in samples]
            out.update({"image": imgs_collated})
    
        if "graph" in samples[0]:
            graph_batches = [Batch.from_data_list(sample["graph"]) for sample in samples]
            out.update({"graph": graph_batches})
        
        return out
    
import numpy as np
from torch.utils.data import Dataset,Subset
from torchvision.datasets import ImageFolder


def datasets_creator(data_dir,transform,labeled_percent=0.1):
    ds = ImageFolder(data_dir,transform=transform)
    labeled_indexes = np.random.choice(len(ds), int(len(ds) * labeled_percent), replace=False)
    labeled_ds = Subset(ds, labeled_indexes)
    semi_labels_ds = SemiCT(ds,labeled_indexes)
    return  labeled_ds,semi_labels_ds
class SemiCT(Dataset):
    def __init__(self,ds,labeled_indexes):
        super(SemiCT, self).__init__()
        self.ds = ds
        self.labeled_indexes = set(labeled_indexes)
        self.labels = []
        self.real_labels = []
        self.features = []
        self.outputs = []

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img,label = self.ds[idx]
        self.real_labels.append(label)
        if idx not in self.labeled_indexes:
            label = -100
        self.labels.append(label)
        return img,label
    def doc(self,features,outputs):
        for i in range(features.shape[0]):
            self.features.append(features[i].unsqueeze(0).cpu().detach())
        for i in range(outputs.shape[0]):
            self.outputs.append(outputs[i].cpu().detach())
        if len(self.labels) > 300:
            self.labels = self.labels[32:]
            self.real_labels = self.real_labels[32:]
            self.features = self.features[32:]
            self.outputs = self.outputs[32:]



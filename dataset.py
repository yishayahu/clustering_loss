from torch.utils.data import Dataset,Subset
def datasets_creator(data_dir,transform,labeled_percent=0.1):
    ds = ImageFolder(data_dir,transform=transform)
    labeled_indexes = set(np.random.choice(len(self.ds), len(self.ds) // labeled_percent, replace=False))
    labeled_ds = Subset(trainset, labeled_indexes)
    semi_labels_ds = SemiCT(ds,labeled_indexes)
    return  labeled_ds,semi_labels_ds
class SemiCT(Dataset):
    def __init__(self,ds,labeled_indexes):
        super(SemiCT, self).__init__()
        self.ds = ds
        self.labeled_indexes = labeled_indexes

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img,label = self.ds[idx]
        if idx not in self.labeled_indexes:
            label = -100
        return img,label

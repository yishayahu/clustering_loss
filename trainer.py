import shutil

import torch.backends.cudnn as cudnn
import torch
import os
import numpy as np
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import datasets_creator, SemiCT
from models import DensNet
from losses import ClusteringLoss



cudnn.benchmark = True


class Trainer(object):
    def __init__(self, cfg, device,exp_name):
        """
        This method initializes settings
        :param settings: application settings
        """

        self.cfg = cfg
        self.images_dir = cfg.IMAGES_DIR
        self.device = device
        self.ckpt_dir = os.path.join(self.cfg.RES_DIR,exp_name)
        os.makedirs(self.ckpt_dir,exist_ok=True)
        shutil.copyfile('config.yml',os.path.join(self.ckpt_dir,'config.yml'))
        self.l1 = [0.1,0.25,0.33,0.5,0.66,0.8,1,1.4,2,2.5,4,6,8]

        t = transforms.Compose([transforms.ToTensor(),transforms.Resize(299),transforms.CenterCrop(256),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.labeled_ds,self.semi_labels_ds = datasets_creator(data_dir=os.path.join(self.images_dir,'train'),transform=t)
        val_ds = ImageFolder(root=os.path.join(self.images_dir,'val'),transform=t)
        self.train_loader = DataLoader(self.labeled_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,drop_last=True)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

        self.net = DensNet(self.cfg.NUM_CLASSES)
        self.net.to(self.device)

        weights = cfg.WEIGHTS
        self.criterion = ClusteringLoss(weights,n_clusters=cfg.N_CLUSTERS,clustering_lambda=self.cfg.CLUSTERING_LOSS_LAMBDA,device=self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.LR)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, cfg.LR_DECAY_STEP_SIZE,
                                                         gamma=cfg.GAMMA_DECAY)
        self.step=0
        wandb.init(
            project="clustering_loss",
            id=wandb.util.generate_id(),
            settings=wandb.Settings(start_method="fork"),
            name=exp_name,
        )

    def visualize(self,ds):
        f= np.concatenate(ds.features)
        out = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(np.concatenate([f,self.criterion.kmeans.cluster_centers_]))
        pred_class_to_shape = {0:'.',1:'^',2:'P',3:'*',4:'x'}
        pred_class_to_arrs = {0:([],[],[]),1:([],[],[]),2:([],[],[]),3:([],[],[]),4:([],[],[])}
        for i in range(out.shape[0]):
            if i >= len(ds.outputs):
                pred = 4
            else:
                pred = torch.argmax(ds.outputs[i]).item()
            X,Y,C = pred_class_to_arrs[pred]
            X.append(out[i][0])
            Y.append(out[i][1])
            if i >= len(ds.outputs):
                C.append('black')
            else:
                if ds.labels[i] == -100:
                    C.append('#88c999' if pred == ds.real_labels[i] else 'hotpink')
                else:
                    C.append('green' if pred == ds.real_labels[i] else 'red')
        for k,(X,Y,C) in pred_class_to_arrs.items():
            plt.scatter(X,Y,c=C,marker=pred_class_to_shape[k])
        im_path = os.path.join(self.ckpt_dir,f'{self.step}.png')
        plt.savefig(im_path)
        wandb.log({'vis':wandb.Image(im_path)})
        plt.cla()
        plt.clf()





    def run_epoch(self,dl,train_or_val):
        bar = tqdm(enumerate(dl), total=len(dl))
        losses = []
        labeled_count = 0
        unlabeled_count = 0
        accs = 0

        for i, (inputs,labels) in bar:
            if train_or_val == 'train':
                self.net.train()  # Set model to training mode
                self.optimizer.zero_grad()
            else:
                self.net.eval()

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            labeled_count += labels[labels!=-100].size(0)
            unlabeled_count += labels[labels==-100].size(0)


            with torch.set_grad_enabled(train_or_val == 'train'):
                outputs,features = self.net(inputs)
                if type(dl.dataset) == SemiCT:
                    dl.dataset.doc(features,outputs)


                _, preds = torch.max(outputs, 1)

                loss = self.criterion(features,outputs, labels)
                if train_or_val == 'train':
                    loss.backward()
                    self.optimizer.step()
                    self.step+=1

            losses.append(loss.item())

            accs += torch.sum(preds[labels!=-100] == labels[labels!=-100]).item()

            if (i % 100 == 99 and train_or_val == 'train') or i == len(dl) -1:
                bar.set_description(f'{train_or_val} loss: {np.mean(losses)} {train_or_val} accuracy: {accs/labeled_count} iter: {i}')
                logs = {
                    f'{train_or_val} loss': float(np.mean(losses)),
                    f'{train_or_val} accuracy': float(accs/labeled_count),
                    f'labeled_p': labeled_count / (labeled_count+unlabeled_count)
                }
                wandb.log(logs,step=self.step)
                if type(dl.dataset) == SemiCT:
                    self.visualize(dl.dataset)


        if train_or_val == 'train':
            self.scheduler.step()
        return float(accs/labeled_count)


    def train(self):
        best_acc = 0.0
        num_epochs = self.cfg.NUM_EPOCHS
        self.step = 0
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            if self.cfg.CLUSTERING_LOSS_LAMBDA > 0 and epoch >= self.cfg.WARMUP_EPOCHS:
                labeled_indexes = self.labeled_ds.indices
                unlabeled_indexes = list(set(list(range(len(self.semi_labels_ds)))) - set(labeled_indexes))
                next_p = self.l1.pop(0)

                ds = torch.utils.data.Subset(self.semi_labels_ds,np.concatenate([np.random.choice(unlabeled_indexes,int(len(labeled_indexes) * next_p) , replace=False), labeled_indexes]))
                self.train_loader = DataLoader(ds, batch_size=self.cfg.BATCH_SIZE, shuffle=True,drop_last=True)

            self.run_epoch(self.train_loader,'train')
            epoch_acc_val = self.run_epoch(self.val_loader,'val')
            if epoch_acc_val > best_acc:
                best_acc = epoch_acc_val

                torch.save({'net': self.net.state_dict()},
                           os.path.join(self.ckpt_dir, 'best_val.pt'))
                torch.save({'optimizer': self.optimizer.state_dict()}, os.path.join(self.ckpt_dir, 'optimizer.pt'))



import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from torch import nn


class ClusteringLoss:
    def __init__(self,weights,n_clusters,clustering_lambda,use_tsne,device):
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(weights,device=device),ignore_index=-100)
        self.kmeans  = MiniBatchKMeans(n_clusters)
        self.device = device
        self.clustering_lambda = clustering_lambda
        self.cache = []
        self.tsne_params = TSNE(n_components=3, learning_rate='auto',init='random')
        self.use_tsne = use_tsne
    def __call__(self, features,outputs, labels):

        l1 = self.ce(outputs,labels)
        if self.clustering_lambda == 0:
            return l1
        l2 = 0.0

        labeled_features = features[labels!=-100]
        unlabeled_features = features[labels==-100]
        if labeled_features.numel() != 0:
            self.cache.append(labeled_features.detach().cpu().numpy())
            if len(self.cache) > 200:
                x = np.concatenate(self.cache)
                if self.use_tsne:
                    x = self.tsne_params.fit_transform(x)
                self.kmeans.partial_fit(x)
                self.cache = []
        if unlabeled_features.shape[0] > 1:
            x= unlabeled_features.detach().cpu().numpy()
            if self.use_tsne:
                x = self.tsne_params.fit_transform(x)
            clusters = self.kmeans.predict(x)
            for i_vec in range(x.shape[0]):
                curr_cluster = self.kmeans.cluster_centers_[clusters[i_vec]]
                l2 += torch.sqrt(torch.cdist(torch.tensor(x[i_vec],device=self.device).unsqueeze(0),torch.tensor(curr_cluster,device=self.device).unsqueeze(0)))
            l2 /= x.shape[0]
        return l1+ self.clustering_lambda * l2

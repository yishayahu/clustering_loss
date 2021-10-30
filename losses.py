import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch import nn


class ClusteringLoss:
    def __init__(self,weights,n_clusters,clustering_lambda,device):
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(weights,device=device),ignore_index=-100)
        self.kmeans  = MiniBatchKMeans(n_clusters)

        self.clustering_lambda = clustering_lambda
        self.cache = []
    def __call__(self, features,outputs, labels):

        l1 = self.ce(outputs,labels)
        if self.clustering_lambda == 0:
            return l1
        l2 = 0.0
        labeled_features = features[labels!=-100]
        unlabeled_features = features[labels==-100]

        if labeled_features.numel() != 0:
            self.cache.append(labeled_features.detach().numpy())
            if len(self.cache) > 30:
                self.kmeans.partial_fit(np.concatenate(self.cache))
                self.cache = []
        if unlabeled_features.numel() != 0:
            clusters = self.kmeans.predict(unlabeled_features.detach().numpy())
            for i_vec in range(unlabeled_features.shape[0]):
                curr_cluster = self.kmeans.cluster_centers_[clusters[i_vec]]
                l2 += torch.cdist(unlabeled_features[i_vec].unsqueeze(0),torch.tensor(curr_cluster).unsqueeze(0))
            l2 /= unlabeled_features.shape[0]
        return l1+ self.clustering_lambda * l2

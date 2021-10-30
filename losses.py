from sklearn.cluster import MiniBatchKMeans

class ClusteringLoss:
    def __init__(self,weights,n_clusters,device):
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(weights,device=device),ignore_index=-100)
        self.kmeans  = MiniBatchKMeans(n_clusters)
    def __call__(self, features,outputs, labels):

        l1 = self.ce(outputs,labels)
        labeled_features = features[labels!=-100]
        unlabeled_features = features[labels==-100]
        self.kmeans.partial_fit(labeled_features)
        clusters = self.kmeans.predict(unlabeled_features)
        l2 = 0.0
        for i_vec in range(unlabeled_features.shape[0])
            curr_cluster = self.kmeans(cluster_centers_[clusters[i_vec]]
            l2 += torch.cdist(unlabeled_features[i_vec].unsqeeuze(0),curr_cluster.unsqeeuze(0))


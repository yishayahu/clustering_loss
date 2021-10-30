def feature
class ModelWrapper(torch.nn.Module):
    def __init__(self):
        super(ModelWrapper, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.features = []
    def forward(self,x):
        self.features = []
        out= self.model(x)
        return out,self.features

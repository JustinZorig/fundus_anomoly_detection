import torch 
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import vgg19, VGG19_Weights
from collections import namedtuple

import torchvision



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#LossOutput = namedtuple("LossOutput",["relu1_2", "relu2_2", "relu3_3", "relu4_3"])

class PerceptualLossNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_model = vgg19(weights=VGG19_Weights.DEFAULT)
        self.vgg_layers = self.vgg_model.features
        
       # self.layer_name_mapping = {
 #     #      '3': "relu1_2",
 #           '8': "relu2_2",
       #     '15':"relu3_3"
#           '22':"relu4_3"
       # }

    
        self.return_nodes = {
                'features.13':"relu6"
        }
 #       self.vgg19_mean = torch.tensor( [0.485,0.456,0.406] ) #RGB)
 #       self.vgg19_std = torch.tensor( [0.229,0.224,0.225] )  #RGB

        self.feature_extractor = create_feature_extractor(self.vgg_model, return_nodes = self.return_nodes).to(device)
        for params in self.feature_extractor.parameters():
            params.requires_grad = False
        del self.vgg_model
        self.loss_fn = nn.MSELoss()


    def forward(self, x, x_hat):
        self.eval()

       # for params in self.feature_extractor.parameters():
       #     print(params.requires_grad)
        
        features = self._feature_extractor(x)
        reconst_features = self._feature_extractor(x_hat)
        loss = self.loss_fn(features['relu6'], reconst_features["relu6"])
        return loss
     

    def _feature_extractor(self, x):
        features = self.feature_extractor(x)
        return features


if __name__ == '__main__':
    loss = PerceptualLossNetwork()
    print(loss.vgg_layers)

    print(torchvision.models.feature_extraction.get_graph_node_names(loss.vgg_model)[0])

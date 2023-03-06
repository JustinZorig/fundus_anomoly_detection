import torch
from torch import nn as nn
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
def to_one_hot(num_classes):
   return lambda y: torch.zeros(num_classes, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)


class EarlyStopper:
    def __init__(self, patience =1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0 
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter =1

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter+=1
            if self.counter >=self.patience:
                return True
        return False
    


def class_weights(images):
  
    count = np.array(list(dict(Counter(images.targets)).values()))
 
    #weights = np.zeros(len(count))
    #for i in range(len(count)):
    #    weights[i] =  1 - (count[i]/sum(count))

    #weights = torch.from_numpy(weights)

    weights = [sum(count) / c for c in count]
    example_weight = [weights[e] for e in [0, 1]]



 
    
    return example_weight



def batch_mean_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean,std
  

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((299,299))
])

dr_train_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/train/train_full_CLAHE',
                                                 transform = transform)
                                               #  target_transform= to_one_hot(dr_num_classes))
dr_train_loader = DataLoader(dr_train_dataset, batch_size =32, shuffle = True, num_workers=20, pin_memory = True)

if __name__ == "__main__":
    mean, std = batch_mean_sd(dr_train_loader)
    print(mean, std)

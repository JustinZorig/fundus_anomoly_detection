import os
import numpy as np
#from skimage import io, transform
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from autoencoder import Autoencoder
from unet import unet
import time
import math
from utils import EarlyStopper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = '/home/miplab/anomoly_detection/preprocessing_justin/'

# hyperparameters
num_epochs =200
batch_size = 100
lr = 0.001
mean = [0.4114,0.2679,0.1820]
std= [0.3016,0.2077,0.1654]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.Normalize(mean, std)
])


train_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/preprocessed/train',
                                                 transform = transform)
train_loader = DataLoader(train_dataset, batch_size =16, shuffle = True, num_workers=20, pin_memory = True)

val_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/preprocessed/validation',
                                                 transform = transform)
val_loader = DataLoader(val_dataset, batch_size =16, shuffle = True, num_workers=20, pin_memory=True)

#test_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/preprocessed/test'), transform = transform)
#test_loader = DataLoader(test_dataset, batch_size=5, shuffle = True, num_workers = 4)
#train_features = next(iter(train_loader))


#model = Autoencoder(input_channels=3, input_resolution=256, hidden_layer_channels=[64,128,256,512],
#                     layer_repetitions=[4,4,4,4], conv_reduced_resolution=16, latent_dimension=200).to(device)
model = unet(input_channels=3, input_resolution=256, hidden_layer_channels=[32,64,128,256,512],#[64,128,256,512,1024],
                     layer_repetitions=[2,2,2,2,1], conv_reduced_resolution=16, latent_dimension=200).to(device)


print(model)


def reconstruction_training_loop():
    early_stopper = EarlyStopper(patience =15, min_delta = 0)
    train_loss_list = []
    val_loss_list = []

    train_num_batches = len(train_loader)
    train_size= len(train_loader.dataset)
    
    val_num_batches = len(val_loader)
    val_size= len(val_loader.dataset)
    
    train_batch_thresh =math.floor(train_num_batches/1 -1)
    val_batch_thresh = math.floor(val_num_batches/1 -1)
  #  start = time.time()
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
       
        model.train()
        for batch_idx, (batch, _) in enumerate(train_loader):
            start_time = time.time()
            batch= batch.to(device)

            batch_loss = model.training_step(batch, batch_idx, epoch)
            train_loss += batch_loss

            if batch_idx%train_batch_thresh==0:
                current = batch_idx*train_loader.batch_size
                ms_per_batch = (time.time() -start_time)*1000
                print(f'| {batch_idx:5d}/{train_num_batches:5d} batches | '
                    f'ms/batch {ms_per_batch:5.2f} | '
                    f'loss {batch_loss:7f} |[{current:>5d}{train_size:>5d}] |')

        model.eval()
        with torch.no_grad():
            for batch_idx, (batch, _) in enumerate(val_loader):
                start_time = time.time()

                batch = batch.to(device)

                loss = model.validation_step(batch, batch_idx, epoch)
                val_loss +=batch_loss
                if batch_idx%val_batch_thresh==0:
                    current = batch_idx*val_loader.batch_size
                    ms_per_batch = (time.time() -start_time)*1000
                    print(f'| {batch_idx:5d}/{val_num_batches:5d} batches | '
                        f'ms/batch {ms_per_batch:5.2f} | '
                        f'loss {batch_loss:7f} |[{current:>5d}{val_size:>5d}] |')
            #print(f'validation loss {loss:7f}')

        print("####################################################################")
       
        train_loss = train_loss/train_num_batches
        val_loss = val_loss/val_num_batches
             
        print("epoch: {}/{}, train loss = {:.6f}   val loss = {:.6f}".format(epoch+1, num_epochs, 
                                    train_loss, val_loss))
        
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)  

        if early_stopper.early_stop(val_loss):
            break

    return train_loss_list, val_loss_list

train_loss, val_loss = reconstruction_training_loop()

torch.save(model.state_dict(), os.path.join(save_path, 'unet_test.pth'))


plt.figure()
plt.plot(train_loss[25:], label='train')
plt.plot(val_loss[25:], label = 'eval')
plt.yscale("log")
plt.legend()
plt.show()


plt.figure()
plt.plot(train_loss, label='train')
plt.plot(val_loss, label = 'eval')
plt.yscale("log")
plt.legend()
plt.show()
#if __name__ == '__main__':
  #  from torch.utils.data import Dataloader
 #   dataset = FundusDataset()
#    dataloader = DataLoader(dataset, batch_size=50, shuffle = True, num_workers=4)
        #for i, batch in enumerate(dataloader):
        

                                    
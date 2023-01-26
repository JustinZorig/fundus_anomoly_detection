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
import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = '/home/miplab/anomoly_detection/preprocessing_justin/'

# hyperparameters
num_epochs =200
batch_size = 100
lr = 0.001


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(128)
])


train_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/preprocessed/train',
                                                 transform = transform)
train_loader = DataLoader(train_dataset, batch_size =8, shuffle = True, num_workers=4)

val_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/preprocessed/validation',
                                                 transform = transform)
val_loader = DataLoader(val_dataset, batch_size =8, shuffle = True, num_workers=4)

#test_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/preprocessed/test'), transform = transform)
#test_loader = DataLoader(test_dataset, batch_size=5, shuffle = True, num_workers = 4)
#train_features = next(iter(train_loader))


model = Autoencoder(base_channel_size=12, latent_dim= 100, input_size =128).to(device)
print(model)

opt_dict = model.configure_optimizers()



def reconstruction_training_loop():
    train_loss_list = []
    val_loss_list = []

    train_num_batches = len(train_loader)
    train_size= len(train_loader.dataset)
    
    val_num_batches = len(val_loader)
    val_size= len(val_loader.dataset)
    
    train_batch_thresh =math.floor(train_num_batches/3 -1)
    val_batch_thresh = math.floor(val_num_batches/2 -1)

    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
       
        model.train()
        for batch, (batch_features, _) in enumerate(train_loader):
            start_time = time.time()
            batch_features = batch_features.to(device)
            opt_dict['optimizer'].zero_grad()

            outputs = model.forward(batch_features)#, batch, epoch)
            model_loss = model._get_reconstruction_loss(batch_features,outputs)
            loss =model_loss.item()
            train_loss += loss

            model_loss.backward() 
            opt_dict["optimizer"].step()
            opt_dict["optimizer"].zero_grad()

            if batch%train_batch_thresh==0:
                current = batch*train_loader.batch_size
                ms_per_batch = (time.time() -start_time)*1000
                print(f'| {batch:5d}/{train_num_batches:5d} batches | '
                    f'ms/batch {ms_per_batch:5.2f} | '
                    f'loss {loss:7f} |[{current:>5d}{train_size:>5d}] |')


        
        model.eval()
        with torch.no_grad():
            for batch, (batch_features, _) in enumerate(val_loader):
                start_time = time.time()

                batch_features = batch_features.to(device)
                opt_dict['optimizer'].zero_grad()

                outputs = model.forward(batch_features)#, batch, epoch)
                model_loss = model._get_reconstruction_loss(batch_features,outputs)
                loss =model_loss.item()
                val_loss +=loss
                if batch%val_batch_thresh==0:
                    current = batch*val_loader.batch_size
                    ms_per_batch = (time.time() -start_time)*1000
                    print(f'| {batch:5d}/{val_num_batches:5d} batches | '
                        f'ms/batch {ms_per_batch:5.2f} | '
                        f'loss {loss:7f} |[{current:>5d}{val_size:>5d}] |')
            #print(f'validation loss {loss:7f}')

            
            #    opt_dict["optimizer"].zero_grad()


        print("####################################################################")
       
        train_loss = train_loss/train_num_batches
        val_loss = val_loss/val_num_batches
        print("epoch: {}/{}, train loss = {:.6f}   val loss = {:.6f}".format(epoch+1, num_epochs, 
                                    train_loss, val_loss))
        
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)    
    return train_loss_list, val_loss_list

train_loss, val_loss = reconstruction_training_loop()

torch.save(model.state_dict(), os.path.join(save_path, 'trained_model2.pth'))



plt.figure()
plt.plot(train_loss, label='train')
plt.plot(val_loss, label = 'eval')
plt.legend()
plt.show()

#if __name__ == '__main__':
  #  from torch.utils.data import Dataloader
 #   dataset = FundusDataset()
#    dataloader = DataLoader(dataset, batch_size=50, shuffle = True, num_workers=4)
        #for i, batch in enumerate(dataloader):
        

                                    
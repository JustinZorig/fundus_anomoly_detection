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
save_path = r'/home/miplab/anomoly_detection/preprocessing_justin/trained_model2.pth'

# hyperparameters
num_epochs=1
batch_size = 100
lr = 0.001


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(128)
])
test_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/preprocessed/test',
                                                 transform = transform)
                                                # target_transform=lambda y: torch.zeros(15,
                                                #  dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
test_loader = DataLoader(test_dataset, batch_size =8, shuffle = True, num_workers=4)


model = Autoencoder(base_channel_size=12, latent_dim= 100, input_size =128).to(device)
model.load_state_dict(torch.load( save_path)) #os.path.join(save_path, 'trained_model2.pth')))
print(model)

opt_dict = model.configure_optimizers()

test_batch_size = 8

#print(test_dataset.classes)
#print(test_dataset.class_to_idx)   #to check how torchvision.datasets encodes the classes


def reconstruction_testing_loop():
    test_loss_list = []
   
    test_num_batches = len(test_loader)
    test_size= len(test_loader.dataset)
    
    test_batch_thresh =math.floor(test_num_batches/3 -1)
    
    for epoch in range(num_epochs):
       
        model.eval()
        with torch.no_grad():
            for batch, (batch_features, _) in enumerate(test_loader):
                print(_)
                test_loss =0
                start_time = time.time()

                batch_features = batch_features.to(device)
                opt_dict['optimizer'].zero_grad()

                outputs = model.forward(batch_features)#, batch, epoch)
                model_loss = model._get_reconstruction_loss(batch_features,outputs)
                loss =model_loss.item()
                test_loss +=loss
                if batch%test_batch_thresh==0:
                    current = batch*test_loader.batch_size
                    ms_per_batch = (time.time() -start_time)*1000
                    print(f'| {batch:5d}/{test_num_batches:5d} batches | '
                        f'ms/batch {ms_per_batch:5.2f} | '
                        f'loss {loss:7f} |[{current:>5d}{test_size:>5d}] |')
                  
                test_loss_list.append(loss)  
            #print(f'validation loss {loss:7f}')

            
            #    opt_dict["optimizer"].zero_grad()


            print("####################################################################")
       
            
            final_test_loss = test_loss/ test_num_batches
            print("epoch: {}/{}, test loss = {:.6f}".format(epoch+1, num_epochs, 
                                    final_test_loss))
      
    return test_loss_list

test_loss = reconstruction_testing_loop()


plt.figure()
plt.plot(test_loss, label='test')
plt.legend()
plt.show()

#if __name__ == '__main__':
  #  from torch.utils.data import Dataloader
 #   dataset = FundusDataset()
#    dataloader = DataLoader(dataset, batch_size=50, shuffle = True, num_workers=4)
        #for i, batch in enumerate(dataloader):
        

                                    
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
save_path = r'/home/miplab/anomoly_detection/preprocessing_justin/resnet_autoencoder_perceptual_loss.pth'

# hyperparameters
num_epochs=1
batch_size = 100
lr = 0.001


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256)
])


test_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/preprocessed/test',
                                                 transform = transform)
test_loader = DataLoader(test_dataset, batch_size =32, shuffle = True, num_workers=20, pin_memory = True)

model = Autoencoder(input_channels=3, input_resolution=256, hidden_layer_channels=[4,4,4,4],
                     layer_repetitions=[4,4,4,4], conv_reduced_resolution=16, latent_dimension=100).to(device)

model.load_state_dict(torch.load( save_path)) #os.path.join(save_path, 'trained_model2.pth')))
print(model)

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
            for batch_idx, (batch, _) in enumerate(test_loader):
               
                test_loss =0
                start_time = time.time()

                batch = batch.to(device)
              
                batch_loss = model.test_step(batch, batch_idx, epoch)
                test_loss +=batch_loss
                if batch_idx%test_batch_thresh==0:
                    current = batch_idx*test_loader.batch_size
                    ms_per_batch = (time.time() -start_time)*1000
                    print(f'| {batch_idx:5d}/{test_num_batches:5d} batches | '
                        f'ms/batch {ms_per_batch:5.2f} | '
                        f'loss {batch_loss:7f} |[{current:>5d}{test_size:>5d}] |')
                  
                test_loss_list.append(batch_loss)  
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
        

                                    
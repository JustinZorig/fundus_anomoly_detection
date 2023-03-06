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
from ffn import ffn_classifier
from utils import to_one_hot, EarlyStopper

import time
import math 
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = r'/home/miplab/anomoly_detection/preprocessing_justin'

# hyperparameters
num_epochs=1
batch_size = 100
lr = 0.001


mean = [0.4114,0.2679,0.1820]
std= [0.3016,0.2077,0.1654]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.Normalize(mean, std)
])

dr_num_classes=2
qual_num_classes=3
dr_test_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/dr_preprocessed_balanced_no_rejects/test',
                                                 transform = transform,
                                                 target_transform= to_one_hot(dr_num_classes))
dr_test_loader = DataLoader(dr_test_dataset, batch_size =32, shuffle = True, num_workers=20, pin_memory = True)

qual_test_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/qual_preprocessed/test',
                                                 transform = transform,
                                                 target_transform= to_one_hot(qual_num_classes))
qual_test_loader = DataLoader(qual_test_dataset, batch_size =32, shuffle = True, num_workers=20, pin_memory=True)



model = Autoencoder(input_channels=3, input_resolution=256, hidden_layer_channels=[4,4,4,4],
                     layer_repetitions=[4,4,4,4], conv_reduced_resolution=16, latent_dimension=200).to(device)

dr_classifier = ffn_classifier(latent_dim=200, num_classes=dr_num_classes).to(device)
qual_classifier = ffn_classifier(latent_dim=200, num_classes=qual_num_classes).to(device)

model.load_state_dict(torch.load( os.path.join(save_path, 'resnet_autoencoder_perceptual_loss_200_guideline.pth'))) #os.path.join(save_path, 'trained_model2.pth')))
dr_classifier.load_state_dict(torch.load(os.path.join(save_path, 'dr_classifier_perceptual_loss_200_batchnorm.pth')))
qual_classifier.load_state_dict(torch.load(os.path.join(save_path, 'qual_classifier_perceptual_loss_200_batchnorm.pth')))


print(model)
print(dr_classifier)
print(qual_classifier)


test_batch_size = 8

#print(test_dataset.classes)
#print(test_dataset.class_to_idx)   #to check how torchvision.datasets encodes the classes



def classifier_testing_loop(classifier, test_loader):

 #   opt_dict = model.configure_optimizers()
 #   classifier_opt_dict = classifier.configure_optimizers()

    y_pred=[]
    y_true=[]
   
    test_num_batches = len(test_loader)
    test_size= len(test_loader.dataset)
    
    test_batch_thresh =math.floor(test_num_batches/3 -1)
    
    for epoch in range(num_epochs):
       
        model.eval()
        classifier.eval()
        
        with torch.no_grad():
            test_loss= 0 
            for batch_idx, (batch, target) in enumerate(test_loader):
    
               # test_loss =0
                start_time = time.time()

                batch = batch.to(device)
                target = target.to(device)

                features= model.obtain_latent_features(batch)#, batch, epoch)
                batch_loss,outputs = classifier.test_step(features, target)

                y_pred.extend( np.argmax(outputs.cpu().numpy(), axis =1))
                y_true.extend( np.argmax(target.cpu().numpy(), axis=1))

                test_loss +=batch_loss
                if batch_idx%test_batch_thresh==0:
                    current = batch_idx*test_loader.batch_size
                    ms_per_batch = (time.time() -start_time)*1000
                    print(f'| {batch_idx:5d}/{test_num_batches:5d} batches | '
                        f'ms/batch {ms_per_batch:5.2f} | '
                        f'loss {batch_loss:7f} |[{current:>5d}/{test_size:>5d}] |') 
   
            print("####################################################################")
       
            
            final_test_loss = test_loss/ test_num_batches
            print("epoch: {}/{}, test loss = {:.6f}".format(epoch+1, num_epochs, 
                                    final_test_loss))
      
    return y_pred, y_true

dr_y_pred, dr_y_true = classifier_testing_loop(dr_classifier, dr_test_loader)
qual_y_pred, qual_y_true = classifier_testing_loop(qual_classifier, qual_test_loader)




dr_classes = ('0', '1')

cf_matrix = confusion_matrix(dr_y_true, dr_y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in dr_classes],
                     columns = [i for i in dr_classes])
plt.figure(figsize = (12,7))
plt.xlabel('actual')
plt.ylabel('predicted')
sn.heatmap(df_cm, annot=True)
plt.show()

print(cf_matrix)
print(classification_report(dr_y_true, dr_y_pred))


classes = ('0', '1', '2')
cf_matrix = confusion_matrix(qual_y_true, qual_y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
plt.xlabel('actual')
plt.ylabel('predicted')
sn.heatmap(df_cm, annot=True)
plt.show()

print(cf_matrix)
print(classification_report(qual_y_true, qual_y_pred))
#if __name__ == '__main__':
  #  from torch.utils.data import Dataloader
 #   dataset = FundusDataset()
#    dataloader = DataLoader(dataset, batch_size=50, shuffle = True, num_workers=4)
        #for i, batch in enumerate(dataloader):
        

                                    
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

def to_one_hot(num_classes):
   return lambda y: torch.zeros(num_classes, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(128)
])

dr_num_classes=2
qual_num_classes=3
dr_test_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/dr_preprocessed/test',
                                                 transform = transform,
                                                 target_transform= to_one_hot(dr_num_classes))
dr_test_loader = DataLoader(dr_test_dataset, batch_size =8, shuffle = True, num_workers=4)

qual_test_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/qual_preprocessed/test',
                                                 transform = transform,
                                                 target_transform= to_one_hot(qual_num_classes))
qual_test_loader = DataLoader(qual_test_dataset, batch_size =8, shuffle = True, num_workers=4)

model = Autoencoder(base_channel_size=12, latent_dim= 100, input_size =128).to(device)

dr_classifier = ffn_classifier(latent_dim=100, num_classes=dr_num_classes).to(device)
qual_classifier = ffn_classifier(latent_dim=100, num_classes=qual_num_classes).to(device)

model.load_state_dict(torch.load( os.path.join(save_path, 'trained_model2.pth'))) #os.path.join(save_path, 'trained_model2.pth')))
dr_classifier.load_state_dict(torch.load(os.path.join(save_path, 'dr_classifier.pth')))
qual_classifier.load_state_dict(torch.load(os.path.join(save_path, 'qual_classifier.pth')))


print(model)
print(dr_classifier)
print(qual_classifier)


test_batch_size = 8

#print(test_dataset.classes)
#print(test_dataset.class_to_idx)   #to check how torchvision.datasets encodes the classes



def classifier_testing_loop(classifier, test_loader):

    opt_dict = model.configure_optimizers()
    classifier_opt_dict = classifier.configure_optimizers()

    test_loss_list = []
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
            for batch, (batch_features, _) in enumerate(test_loader):
    
               # test_loss =0
                start_time = time.time()

                batch_features = batch_features.to(device)
                _=_.to(device)

                features= model.obtain_latent_features(batch_features)#, batch, epoch)
                outputs= classifier.forward(features)
           #     print(outputs)
             #   fsdfsd = np.argmax(outputs.cpu().numpy(), axis=1)
            #    print(fsdfsd)
                y_pred.extend( np.argmax(outputs.cpu().numpy(), axis =1))
                y_true.extend( np.argmax(_.cpu().numpy(), axis=1))

                classifier_loss = classifier._get_loss(outputs, _)
                
                loss =classifier_loss.item()
                test_loss +=loss
                if batch%test_batch_thresh==0:
                    current = batch*test_loader.batch_size
                    ms_per_batch = (time.time() -start_time)*1000
                    print(f'| {batch:5d}/{test_num_batches:5d} batches | '
                        f'ms/batch {ms_per_batch:5.2f} | '
                        f'loss {loss:7f} |[{current:>5d}/{test_size:>5d}] |')
                  
                test_loss_list.append(loss)  
   
            print("####################################################################")
       
            
            final_test_loss = test_loss/ test_num_batches
            print("epoch: {}/{}, test loss = {:.6f}".format(epoch+1, num_epochs, 
                                    final_test_loss))
      
    return test_loss_list, y_pred, y_true

dr_test_loss, dr_y_pred, dr_y_true = classifier_testing_loop(dr_classifier, dr_test_loader)
qual_test_loss, qual_y_pred, qual_y_true = classifier_testing_loop(qual_classifier, qual_test_loader)


plt.figure()
plt.plot(dr_test_loss, label='test')
plt.legend()
plt.show()

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




plt.figure()
plt.plot(qual_test_loss, label='test')
plt.legend()
plt.show()

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
        

                                    
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
from utils import to_one_hot, EarlyStopper, class_weights

import time
import math 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = r'/home/miplab/anomoly_detection/preprocessing_justin'

# hyperparameters
num_epochs=50
batch_size = 100
lr = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256)
])

############################# DR
dr_classes_num =2

dr_train_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/dr_preprocessed_balanced_no_rejects/train',
                                                 transform = transform,
                                                 target_transform= to_one_hot(dr_classes_num))
dr_val_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/dr_preprocessed_balanced_no_rejects/validation',
                                                 transform = transform,
                                                 target_transform= to_one_hot(dr_classes_num))

#dr_weights = class_weights(dr_train_dataset)
#dr_weights = [1.0, 7.0]
#dr_weights = torch.DoubleTensor(dr_weights)
#dr_sampler = torch.utils.data.sampler.WeightedRandomSampler(dr_weights, dr_classes_num)


# Quick note: WeightedRandomSampler already shuffles data so we must remove the shuffle argument in the DataLoaders
dr_train_loader = DataLoader(dr_train_dataset, batch_size =32, shuffle=True, num_workers=20, pin_memory = True)
dr_val_loader = DataLoader(dr_val_dataset, batch_size =32, shuffle= True, num_workers=20, pin_memory=True)


############################# Quality
qual_classes_num = 3
qual_train_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/qual_preprocessed/train',
                                                 transform = transform,
                                                 target_transform= to_one_hot(qual_classes_num))
qual_val_dataset = torchvision.datasets.ImageFolder(root = '/home/miplab/data/Kaggle_Eyepacs/qual_preprocessed/validation',
                                                 transform = transform,
                                                 target_transform= to_one_hot(qual_classes_num))

qual_weights = class_weights(qual_train_dataset)
qual_weights = torch.DoubleTensor(qual_weights)
qual_sampler = torch.utils.data.sampler.WeightedRandomSampler(qual_weights, qual_classes_num)


qual_train_loader = DataLoader(qual_train_dataset, batch_size =32, shuffle = True, num_workers=20, pin_memory=True)
qual_val_loader = DataLoader(qual_val_dataset, batch_size =32, shuffle = True, num_workers=20, pin_memory=True)


model = Autoencoder(input_channels=3, input_resolution=256, hidden_layer_channels=[4,4,4,4],
                     layer_repetitions=[4,4,4,4], conv_reduced_resolution=16, latent_dimension=100).to(device)

model.load_state_dict(torch.load( os.path.join(save_path, 'resnet_autoencoder.pth')))



dr_classifier = ffn_classifier(latent_dim =100, num_classes=dr_classes_num, dropout_rate = 0.3).to(device)
qual_classifier = ffn_classifier(latent_dim=100, num_classes=qual_classes_num, dropout_rate = 0.3).to(device)


#print(model)
print(dr_classifier)
print(qual_classifier)



#print(test_dataset.classes)
#print(test_dataset.class_to_idx)   #to check how torchvision.datasets encodes the classes


def classifier_training_loop(classifier, train_loader, val_loader):
    early_stopper = EarlyStopper(patience = 10, min_delta=10)

    opt_dict = model.configure_optimizers()
    classifier_opt_dict = classifier.configure_optimizers()

    train_loss_epoch = []
    val_loss_epoch = []
    learning_rate = []

    train_num_batches = len(train_loader)
    train_size= len(train_loader.dataset)
    
    val_num_batches = len(val_loader)
    val_size= len(val_loader.dataset)
    
    train_batch_thresh =math.floor(train_num_batches/2 -1)
    val_batch_thresh = math.floor(val_num_batches/2 -1)

    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
       
        model.eval()
        classifier.train()
        for batch, (batch_features, _) in enumerate(train_loader):
            start_time = time.time()

            batch_features = batch_features.to(device)
            _ = _.to(device)

            opt_dict['optimizer'].zero_grad()  
            classifier_opt_dict['optimizer'].zero_grad()

            features = model.obtain_latent_features(batch_features)#, batch, epoch)
            outputs = classifier(features)
            
            classifier_loss = classifier._get_loss(outputs, _)
            loss =classifier_loss.item()
            train_loss += loss

            classifier_loss.backward() 
            classifier_opt_dict["optimizer"].step()

            if batch%train_batch_thresh==0:
                current = batch*train_loader.batch_size
                ms_per_batch = (time.time() -start_time)*1000
                print(f'| {batch:5d}/{train_num_batches:5d} batches | '
                    f'ms/batch {ms_per_batch:5.2f} | '
                    f'loss {loss:7f} |[{current:>5d}/{train_size:>5d}] |'
                    f'learning rate {classifier_opt_dict["optimizer"].param_groups[0]["lr"]:7f}')

        
        model.eval()
        classifier.eval()
        with torch.no_grad():
            for batch, (batch_features, _) in enumerate(val_loader):
                start_time = time.time()

                batch_features = batch_features.to(device)
                _=_.to(device)

                features = model.obtain_latent_features(batch_features)#, batch, epoch)
                outputs = classifier(features)
                classifier_loss = classifier._get_loss(outputs, _)


                loss =classifier_loss.item()
                val_loss +=loss
                if batch%val_batch_thresh==0:
                    current = batch*val_loader.batch_size
                    ms_per_batch = (time.time() -start_time)*1000
                    print(f'| {batch:5d}/{val_num_batches:5d} batches | '
                        f'ms/batch {ms_per_batch:5.2f} | '
                        f'loss {loss:7f} |[{current:>5d}/{val_size:>5d}] |'
                        f' learning rate {classifier_opt_dict["optimizer"].param_groups[0]["lr"]}')

        #        if early_stopper.early_stop(loss):
        #            break
  

        print("####################################################################")


        train_loss = train_loss/train_num_batches
        val_loss = val_loss/val_num_batches


        learning_rate.append(classifier_opt_dict["optimizer"].param_groups[0]["lr"])
        classifier_opt_dict["lr_scheduler"].step(val_loss)

        print("epoch: {}/{}, train loss = {:.10f}   val loss = {:.10f}".format(epoch+1, num_epochs, 
                                    train_loss, val_loss))
        
        train_loss_epoch.append(train_loss)
        val_loss_epoch.append(val_loss)  

        if early_stopper.early_stop(val_loss):
            break

    return train_loss_epoch, val_loss_epoch, learning_rate

dr_train_loss, dr_val_loss ,dr_lr= classifier_training_loop(dr_classifier, dr_train_loader, dr_val_loader)
qual_train_loss, qual_val_loss, qual_lr = classifier_training_loop(qual_classifier, qual_train_loader, qual_val_loader)

torch.save(dr_classifier.state_dict(), os.path.join(save_path, 'dr_classifier.pth'))
torch.save(qual_classifier.state_dict(), os.path.join(save_path, 'qual_classifier.pth'))


plt.figure()
plt.plot(dr_train_loss, label='train')
plt.plot(dr_val_loss, label='val')
plt.legend()
plt.show()


plt.figure()
plt.plot(qual_train_loss, label='train')
plt.plot(qual_val_loss, label='val')
plt.legend()
plt.show()

plt.figure()
plt.plot(dr_lr, dr_train_loss, label='train')
plt.plot(dr_lr, dr_val_loss, label='val')
plt.xlabel("learning rate")
plt.legend()
plt.show()


plt.figure()
plt.plot(qual_lr, qual_train_loss, label='train')
plt.plot(qual_lr, qual_val_loss, label='val')
plt.ylabel("learning rate")
plt.legend()
plt.show()

#if __name__ == '__main__':
  #  from torch.utils.data import Dataloader
 #   dataset = FundusDataset()
#    dataloader = DataLoader(dataset, batch_size=50, shuffle = True, num_workers=4)
        #for i, batch in enumerate(dataloader):
        

                                    
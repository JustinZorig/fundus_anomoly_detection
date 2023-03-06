from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import classification_report, confusion_matrix
from utils import EarlyStopper, to_one_hot, set_parameter_requires_grad, initialize_model, create_loss, create_scheduler
from torch.utils.data.sampler import WeightedRandomSampler
import seaborn as sn
import pandas as pd
from collections import Counter

 
def train_model(model, dataloaders, criterion, optimizer, scheduler=None,  num_epochs = 25, is_inception = False):
    early_stopper = EarlyStopper(patience =10, min_delta = 0.001)

    since = time.time()
    val_acc_history = []
    train_acc_history = []
    val_loss_history = []
    train_loss_history = []

  
    best_acc = 0.0
    best_loss = np.inf 
    best_model_wts = copy.deepcopy(model.state_dict())
    

    for epoch in range(num_epochs):
        print('Epoch {}/{}'. format(epoch, num_epochs-1))
        print('-'*10)

        stop_bool= False
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.

                    outputs = model(inputs)
                
                    if is_inception and phase == 'train':
                        #outputs, aux_outputs = model(inputs)
                        _, preds = torch.max(outputs[0], 1) 
                        loss1 = criterion(outputs[0],labels)
                        loss2 = criterion(outputs[1], labels)
                        loss = loss1 +0.4*loss2

                    else:
                        #outputs = model(inputs)
                        # Out of self interest, the scope of a variable initialized/assigned within an if/else block 
                        # is within the entire function/class/etc.   
                        _, preds = torch.max(outputs, 1)         
                        loss = criterion(outputs, labels)   # crossentropy between output logits, and target labels
                       
                    # backward and optimize only if traiining
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                _, labels = torch.max(labels,1)
                running_corrects += torch.sum(preds == labels.data)
               

            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss : {:.7f} Acc: {:.7f}'.format(phase, epoch_loss, epoch_acc))

          #  if phase =='val' and epoch_acc >best_acc:
          #      best_acc = epoch_acc
          #      best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc

                val_acc_history.append(epoch_acc.cpu())
                val_loss_history.append(epoch_loss)
                stop_bool =early_stopper.early_stop(epoch_loss)
                if epoch_loss <best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                if stop_bool:
                    break

            if phase == 'train':
                train_acc_history.append(epoch_acc.cpu())
                train_loss_history.append(epoch_loss)
        if stop_bool:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best_val_acc {:4f}'.format(best_acc))

    #model.load_state_dict(best_model_wts)

    return model, val_acc_history, train_acc_history, val_loss_history, train_loss_history, best_model_wts



def test_model(model, dataloaders, criterion, optimizer, is_inception = False):
    """
    By not shuffling the test dataset, we can also obtain the filenames of the samples
    in the correct order that each iamge tensor appears in during batch iteration. To 
    do this, we use the "imgs" attribute of the dataset.

    """
    since = time.time()
    test_acc_history = []

  #  best_model_wts = copy.deepcopy(model.state_dict())
  #  bes_acc = 0.0

    model.eval()
    y_pred = []
    y_true = []
    filenames =  [x[0] for x in dataloaders.dataset.imgs]
    
    model.eval()    
    with torch.no_grad():
        for inputs, labels  in dataloaders:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            _, labels = torch.max(labels, 1)
            y_pred.extend( preds.cpu().numpy())
            y_true.extend( labels.cpu().numpy())

                    # backward and optimize only if traiining
             
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))

    return model, test_acc_history, y_pred, y_true, filenames


 #####################################################################################



def class_weights(images, batch_size):
    count = np.array(list(dict(Counter(images.targets)).values()))
    #weights = [sum(count) / c for c in count]
    weights = [1.0 / c for c in count]
    #assign weight to each sample! 
    samples_weight = np.array([weights[e] for e in images.targets]) 

    
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    # replacement = True necessary to actually perform oversampling of minority class!
    sampler= WeightedRandomSampler(samples_weight,len(samples_weight) , replacement = True)
    return sampler


    



if __name__ =="__main__":

    training= True
    testing = True
    filtered= True

    model_name = "inception"
    loss_type ="bce"

    num_classes = 2
    batch_size = 64
    num_epochs = 100
    feature_extract = False # Only extracting features and training the classification head.

    mean = [0.3238]
    std= [0.2581]
 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_path = '/home/miplab/anomoly_detection/transfer_learning'

    if filtered:
        data_dir = "/home/miplab/data/Kaggle_Eyepacs/EyeQ/EyeQ_dr/CLAHE/good_only"
        test_dir = "/home/miplab/data/Kaggle_Eyepacs/EyeQ/EyeQ_dr/CLAHE/good_only/test"

        model_saved_file = 'finetuned_inception_SGD_good_only_EyeQ.pth'
        predictions_save_filename = 'dr_predictions_good_only_EyeQ.csv'
    else:
        data_dir = "/home/miplab/data/Kaggle_Eyepacs/EyeQ/EyeQ_dr/CLAHE/full"
        test_dir = "/home/miplab/data/Kaggle_Eyepacs/EyeQ/EyeQ_dr/CLAHE/full/test"

        model_saved_file = 'finetuned_inception_SGD_full_EyeQ.pth'
        predictions_save_filename = 'dr_predictions_full_EyeQ.csv'

  

 #   if filtered:
 #       data_dir = "/home/miplab/data/Kaggle_Eyepacs/train/filtered_CLAHE/2_class"
 #       test_dir ="/home/miplab/data/Kaggle_Eyepacs/test/filtered_CLAHE_full/2_class"

 #       model_saved_file = 'finetuned_inception_SGD_filtered_trainset.pth'
 #       predictions_save_filename = 'eyepacs_dr_predictions_filtered_testset.csv'
 #   else:
 #       data_dir = "/home/miplab/data/Kaggle_Eyepacs/train/CLAHE/2_class"
 #       test_dir ="/home/miplab/data/Kaggle_Eyepacs/test/test_full_CLAHE/2_class"

 #       model_saved_file = 'finetuned_inception_SGD_full_trainset.pth'
 #       predictions_save_filename = 'eyepacs_dr_predictions_full_testset.csv'


    training_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize((299,299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
    ])
    
    testing_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize((299,299))
    ])

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract)
    print(model_ft)
    model_ft = model_ft.to(device) 

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update =[]
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # send model to gpu
    # observe all paramets aer being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum = 0.9,weight_decay = 0.000006, nesterov = True)
    #optimizer_ft = optim.Adam(params_to_update,lr=0.0001, weight_decay = 0.00005)

    criterion = create_loss(loss_type)
   

    if training:
             
        image_datasets = {'train' : datasets.ImageFolder(root = os.path.join(data_dir, 'train'), transform = training_transform,
                                                    target_transform= to_one_hot(num_classes)),

                        'val': datasets.ImageFolder(root = os.path.join(data_dir, 'val'), transform = testing_transform,
                                                    target_transform= to_one_hot(num_classes))}

                                   
        sampler = class_weights(image_datasets['train'], batch_size)  
        
        dataloaders_dict = {'train' : torch.utils.data.DataLoader(image_datasets['train'] , batch_size=batch_size, sampler = sampler,
                                                     num_workers=16, pin_memory = True,  drop_last = True),
                            'val' : torch.utils.data.DataLoader(image_datasets['val'] , batch_size=batch_size, 
                                                     num_workers=16, pin_memory = True)}

        scheduler = create_scheduler(optimizer_ft, dataloaders_dict['train'], num_epochs)     
                                                      
     
        model_ft, val_acc_hist, train_acc_hist, val_loss_hist, train_loss_hist, best_model_wts = train_model(model_ft, dataloaders_dict, 
                criterion, optimizer_ft, scheduler, num_epochs = num_epochs, is_inception = (model_name == "inception"))

        torch.save(best_model_wts, os.path.join(save_path, model_saved_file))

        plt.figure()
        plt.plot(train_loss_hist, label='train loss')
        plt.plot(val_loss_hist, label='val')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(train_acc_hist, label='train acc')
        plt.plot(val_acc_hist, label='val')
        plt.legend()
        plt.show()

    if testing:
        print("Starting testing")
        dr_test_dataset = torchvision.datasets.ImageFolder(root = os.path.join(test_dir),transform = testing_transform, 
                                            target_transform = to_one_hot(num_classes))
        dr_test_loader = torch.utils.data.DataLoader(dr_test_dataset, batch_size =batch_size, 
                                            num_workers=16, pin_memory = True)

        model_ft.load_state_dict(torch.load( os.path.join(save_path,model_saved_file)))
        model_ft, test_acc, y_pred, y_true, filenames = test_model(model_ft, dr_test_loader, criterion, optimizer_ft)

        df = pd.DataFrame(list(zip(filenames, y_true, y_pred)), columns = ['images','dr ground truth', 'dr prediction'])
        df.to_csv(os.path.join(save_path, predictions_save_filename))
        dr_classes = ('0', '1' )#, '2', '3', '4')

        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in dr_classes],
                     columns = [i for i in dr_classes])
        plt.figure(figsize = (12,7))
        plt.xlabel('actual')
        plt.ylabel('predicted')
        sn.heatmap(df_cm, annot=True)
        plt.show()

        print(cf_matrix)
        print(classification_report(y_true, y_pred))



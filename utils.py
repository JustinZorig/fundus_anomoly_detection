import torch
from torch import nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch.distributions
import torch.optim as optim


def to_one_hot(num_classes):
   return lambda y: torch.zeros(num_classes, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)

def to_ordinal(num_classes):
    return lambda y: torch.zeros(num_classes, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)   

def ordinal_prediction_to_label(pred):
    """Convert ordinal predictions to class labels, e.g.
    
    [0.9, 0.1, 0.1, 0.1] -> 0
    [0.9, 0.9, 0.1, 0.1] -> 1
    [0.9, 0.9, 0.9, 0.1] -> 2
    etc.
    """
    return torch.sum((pred>0.5), dim=1 )

def lr_scheduler_test(model, dataloaders, criterion, optimizer, scheduler=None,  num_epochs = 25):
    """
    Used to quickly check that the scheduler adjusts the learning rate acording
    to the anticipated plot
    """
    count_val=[0,0]
    count_train=[0,0]

    epochs =25
    learning_rates=[]
    total_steps = epochs* len(dataloaders['train'])

    for epoch in range(epochs):
        
        phase= 'train'
        for inputs, labels in dataloaders['train']:

            if phase =='train':
                scheduler.step()
                learning_rates.append(scheduler.get_last_lr())
        print("epoch {} out of {}".format(epoch,epochs))

    print(total_steps)
    print(len(learning_rates))
    plt.figure()
    plt.plot(learning_rates)
    plt.show()


def sampler_test(dataloaders, num_epochs ):
    """
    Used to quickly check that the weighted random sampler is performing
    under/over sampling to the training data to ensure a more balanced 
    set.
    """
    count_val=[0,0]
    count_train=[0,0]

    epochs =25
    for epoch in range(epochs):
        for phase in ['train', 'val']:

            for inputs, labels in dataloaders[phase]:
                
                _, labels = torch.max(labels,1)
               
                labels = labels.tolist()
               
                if phase == 'train':

                    count_train[0] += labels.count(0)
                    count_train[1] += labels.count(1)
                else:
                    count_val[0] += labels.count(0)
                    count_val[1] += labels.count(1)

               # if phase=='train':

    print(count_train)                
    print(count_val)
    

class EarlyStopper:
    def __init__(self, patience =1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0 
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter =0

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter+=1
            if self.counter >=self.patience:
                return True
        return False
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def create_loss(loss_type:str):
    if loss_type =="ordinal":
        return nn.MSELoss()
    elif loss_type == "crossentropy":
        return nn.CrossEntropyLoss()
    else:
         return nn.BCEWithLogitsLoss()

def create_scheduler(optimizer, dataloader, num_epochs:int):
    """
    Learning rate will increase from max_lr/div_factor to max_lr in the first (pct_start*total_steps) steps
    Then learning rate will decrease from max_lr to max_lr/final_div_factor

    """
    div_factor = 25 # default is 25
    final_div_factor = 1e4 # default is 1e4
    pct_start =0.1 # percentage of total steps it takes to increase to max_lr
    num_epochs =25
    
    total_steps = num_epochs * len(dataloader)# num_epochs * num_batches per Epoch
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps = total_steps, pct_start = pct_start)
    return scheduler


def initialize_model(model_name, num_classes, feature_extract, use_pretrained = True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """Resnet18
        """ 
        model_ft = models.resnet18(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "densenet":
        """Dense net
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg19":
        model_weights = models.VGG19_Weights
        model_ft = models.vgg19(weights = model_weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 256
    
    elif model_name == "inception":
        """
        inception v3: expects 299,299 sized image and has an auxiliary output
        """

        model_weights = "DEFAULT"
        model_ft = models.inception_v3(weights= model_weights)
        set_parameter_requires_grad(model_ft, feature_extract)

        # handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        #Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size =299

    
    return model_ft, input_size    

class UncertaintyNetwork(nn.Module):
    #https://romainstrock.com/blog/modeling-uncertainty-with-pytorch.html
    def __init__(self, n_hidden, classifier):
        super().__init__()

        self.classifier = classifier

        self.mean_layer = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(n_hidden,1),
        )

        self.std_layer = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(n_hidden,1),
                nn.Softplus(), # enforces positiivity
        )

    def forward(self, x):
        embedding = self.classifier(x)

        # parameterizatio of the mean
        mean = self.mean_layer(embedding)
        stdev = self.std_layer(embedding)

        return torch.distributions.Normal(mean, stdev)
    
    def compute_loss(model,x,y):
        normal_dist = model(x)
        neg_log_likelihood = -normal_dist.log_prob(y)
        return torch.mean(neg_log_likelihood)



if __name__ == "__main__":
    stoper = EarlyStopper(5,0.1)

    val_loss_list= [2, 1.9, 2.8, 1.7, 1.6, 1.5, 1, 1.1, 1,3,1.5, 1.7, 1.7, 1.9, 1.2,2,3,23,234,132,1.2]

    for a in val_loss_list:
        if stoper.early_stop(a):
            break

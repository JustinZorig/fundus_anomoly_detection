import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ffn_classifier(nn.Module):
    def __init__(self,
                latent_dim: int,
                num_classes: int,
                dropout_rate: int = 0.0):

        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = self._create_network()
        self.opt_dict = self._configure_optimizers()

    def _create_network(self):
        layers = [nn.Linear(self.latent_dim, 50),
                nn.ReLU(),
                nn.BatchNorm1d(50),
                nn.Dropout(self.dropout_rate),
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.BatchNorm1d(25),
                nn.Dropout(self.dropout_rate),
                nn.Linear(25, self.num_classes),
                nn.ReLU()
                ]
        self.net = nn.Sequential(*layers)
    
    def _configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = 1e-3)

        # Using a scheduler is optional but can be helpful
        # Scheduler reduces lr if validation performance hasn't imporved for the last n epochs

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode = 'min',
                                                        factor = 0.5,
                                                        patience =10,
                                                        min_lr = 5e-6)                                               
        opt_dict = {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return opt_dict


    def forward(self, x):
        x = self.net(x)
        return x
    
    def _get_loss(self, y_hat, y):
        """
        Given a batch of FFN outputs (y_hat) and true labels,
        this function returns the classification loss
        """

        loss = self.loss_fn(y_hat, y)
        loss =loss.mean(dim=[0])
        return loss

    def training_step(self, x,y):        

        self.opt_dict["optimizer"].zero_grad()  # zero gradients for every batch

        y_hat = self(x)                       #prediction
        loss = self._get_loss(y_hat, y)       #get loss

        loss.backward()                       #compute gradient
        self.opt_dict["optimizer"].step()     #update model parameter based on current gradient

    #    loss = loss.mean(dim=[0]).item()        # Avg the loss across the batch
 #       self.log('train_loss', loss)

        return loss.item()

    def validation_step(self, x, y):
        y_hat = self(x)
        loss = self._get_loss(y_hat, y)
      #  loss = loss.mean(dim=[0]).item()  # Avg the loss across the batch
 #       self.log('val_loss', loss)
        return loss.item()
    
    def test_step(self, x, y):
        y_hat = self(x)
        loss = self._get_loss(y_hat, y)
      #  loss = loss.mean(dim=[0]).item()  # Avg the loss across the batch
 #       self.log('test_loss, loss')
        return loss.item(), y_hat

    
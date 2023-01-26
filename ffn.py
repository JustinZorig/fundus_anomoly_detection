import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ffn_classifier(nn.Module):
    def __init__(self,
                latent_dim: int,
                num_classes: int):

        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 50) # in_features, out_features
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25,num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    

    def forward(self, x):
        x= self.relu(self.fc1(x))
        x= self.dropout(x)
        x= self.relu(self.fc2(x))
        x= self.dropout(x)
        x= self.relu(self.fc3(x))
        return x
    
    
    def _get_loss(self, output, target):
        """
        Given a batch of FFN outputs (y_hat) and true labels,
        this function returns the classification loss
        """
        


        loss = F.cross_entropy(output, target, reduction = "none") # input, target
        loss = loss.mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = 1e-3)

        # Using a scheduler is optional but can be helpful
        # Scheduler reduces lr if validation performance hasn't imporved for the last n epochs

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode = 'min',
                                                        factor = 0.2,
                                                        patience =20,
                                                        min_lr = 5e-5)
        ret_dict = {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

        return (ret_dict)

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss, loss')
    
    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        sel.log('test_loss, loss')

    
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

class Encoder(nn.Module):
    def __init__(self, 
                num_input_channels: int,
                base_channel_size : int,
                latent_dim: int,
                input_size: int,
                act_fn: object = nn.GELU):

        """ 
        Inputs:
            - num_input_channel: Number of channels of input image.
            - base_channel_size: Number of channels in the first conv layers.
            - latent_dim: Dimensionality of latent representations.
            - act_fn: Activation function used throughout the encoder network.
        """

        super().__init__()
        c_hid = base_channel_size
        self.size = int(input_size/8)
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding =1, stride =2),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Flatten(),
            nn.Linear(2*self.size*self.size*c_hid, latent_dim)
        )

    def forward(self, x):
        return self.net(x)



class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 input_size:int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.size = int(input_size/8)
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*self.size*self.size*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, self.size, self.size)
        x = self.net(x)
        return x       



class Autoencoder(nn.Module):
    def __init__(self,
                base_channel_size: int,
                latent_dim: int,
                input_size: int,
                encoder_class: object = Encoder,
                decoder_class: object = Decoder,
                num_input_channels: int =3,
                width: int = 32,
                height: int =32):
    
        super().__init__()
        # saving hyperparameters of autoencoder
       # self.save_hyperparameters()
        #Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim, input_size)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim, input_size)
        # Example input array needed for visualizing network graph
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)


    def obtain_latent_features(self, x):
        z = self.encoder.forward(x)        
        return z

    def forward(self, x):#, batch_num, epoch):
        """
        The forward function takes in an image and returns the
        reconstructed image/
        """
        z = self.encoder.forward(x)
        x_hat = self.decoder.forward(z)
   
      #  if(batch_num==2500) and epoch ==198:

            #xx_hat= np.transpose(x_hat[0].cpu().detach().numpy(), (1,2,0))
            #xx =np.transpose(x[0].cpu().detach().numpy(), (1,2,0))
        
            #plt.figure()
            #plt.imshow(xx_hat)
            #plt.show()

            #plt.figure()
            #plt.imshow(xx)
            #plt.show()
            
        return x_hat

    def _get_reconstruction_loss(self, batch, x_hat):
        """
        Given a batch of images, this function retursn the reconstruction 
        loss (MSE in our case
        """

        x = batch  # we do not need labels
        #x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction = "none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
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

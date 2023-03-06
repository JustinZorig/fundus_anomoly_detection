import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from loss_function import PerceptualLossNetwork
from contracter import ContractingNet
from expander import ExpandingNet

class unet(nn.Module):
    def __init__(self,
                input_channels: int,
                input_resolution,
                hidden_layer_channels: list,
                layer_repetitions: list,
                conv_reduced_resolution,
                latent_dimension:int,

                encoder_class: object = ContractingNet,
                decoder_class: object = ExpandingNet,
                ):
    
        super().__init__()
        # saving hyperparameters of autoencoder
       # self.save_hyperparameters()
        #Creating encoder and decoder

        self.encoder = encoder_class(input_channels, input_resolution,  hidden_layer_channels, layer_repetitions,
                        conv_reduced_resolution, latent_dimension)
        hidden_layer_channels.reverse()  # need to reverse order of decoder part 
        layer_repetitions.reverse()

        self.decoder = decoder_class(input_channels, input_resolution, hidden_layer_channels, layer_repetitions,
                        conv_reduced_resolution, latent_dimension)

        #self.loss_fn = nn.MSELoss(reduction="none")
        self.loss_fn = PerceptualLossNetwork()
        
        self.opt_dict = self._configure_optimizers()
        


    def obtain_latent_features(self, x):
        z = self.encoder(x)        
        return z

    def forward(self, x, batch_num, epoch):
        """
        The forward function takes in an image and returns the
        reconstructed image/
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
   

      #  if(batch_num==274) and (epoch == 199):
      #  if batch_num == :
      #      xx_hat= np.transpose(x_hat[0].cpu().detach().numpy(), (1,2,0))
      #      xx =np.transpose(x[0].cpu().detach().numpy(), (1,2,0))
        
      #      plt.figure()
      #      plt.imshow(xx_hat)
      #      plt.show()

      #      plt.figure()
      #      plt.imshow(xx)
      #      plt.show()
            
        return x_hat

    def _get_reconstruction_loss(self, x, x_hat):
        """
        Given a batch of images, this function retursn the reconstruction 
        loss (MSE in our case
        """

        #loss = self.loss_fn(x, x_hat)#, reduction = "none")
        #loss = loss.sum(dim=[1,2,3]).mean(dim=[0])

        loss = self.loss_fn(x, x_hat)
        return loss

    def _configure_optimizers(self):
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

    def training_step(self, x, batch_num, epoch):
        self.opt_dict["optimizer"].zero_grad()
        x_hat = self(x, batch_num, epoch)
        loss = self._get_reconstruction_loss(x, x_hat)
        loss.backward()
        self.opt_dict["optimizer"].step()
    
    #    self.log('train_loss', loss)
        return loss.item()

    def validation_step(self, x, batch_num, epoch):
        x_hat = self(x, batch_num, epoch)
        loss = self._get_reconstruction_loss(x, x_hat)
    #    self.log('val_loss, loss')
        return loss.item()
    
    def test_step(self, x, batch_num, epoch):
        x_hat = self(x,batch_num,epoch)
        loss = self._get_reconstruction_loss(x,x_hat)
    #    self.log('test_loss, loss')
        return loss.item()

    def visualize_reconsturctions(self,x):
        self.eval()
        with torch.no_grad():
            reconst_imgs = self(x)
        reconst_imgs=reconst_imgs.cpu()

        # Plotting
        imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
        grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, range=(-1,1))
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(7,4.5))
        plt.title(f"Reconstructed from {model.params.latent_dim} latents")
        plt.imshow(grid)
        plt.axis('off')
        plt.show()








if __name__ == '__main__':
    x = unet(3,256, [64,128,256,512], [3,3,3,3],16, 100)
    print(x)
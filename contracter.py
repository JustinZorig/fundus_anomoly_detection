import torch
import torch.nn as nn




class ContractingNet(nn.Module):
    def __init__(self,
                input_channels,
                input_resolution,

                hidden_layer_channels: list,
                layer_repetitions: list,

                conv_output_resolution,
                latent_dimension ):
        """
        """
        super().__init__()

        self.input_channels = input_channels
        self.input_resolution = input_resolution
        self.hidden_channels = hidden_layer_channels
        self.layer_repetitions = layer_repetitions

        self.conv_output_resolution = conv_output_resolution
        self.latent_dimension = latent_dimension
        self.trace =[]
        self.skip_idxs =[]
        self.model = self._create_network()

    def _create_network(self):
    
        layers = [ConvBlock(self.input_channels, self.hidden_channels[0], 3, 1, 1)]
        skip_idx=0

        for channels, repeat_val, idx in zip(self.hidden_channels, self.layer_repetitions, range(len(self.hidden_channels))):
            layers_temp = []
            for block_num in range(repeat_val):
                layers += [ResnetBlock(channels, 3, 1, 1, use_dropout = True)]
                skip_idx+=1

            #we make the last resnet per block downsample  height and width, increase channels. DO not downsample the for the last channel
            if channels!= self.hidden_channels[-1]:      
                layers += [DownConv(channels, self.hidden_channels[idx+1], 3, 1, 1)]
            # we append before iterating the inclusion of the downconv because the skip connections will be the feature maps
            # prior to downsampling. Needs to be concatenated with the expander feature maps AFTER the upconv
                self.skip_idxs.append(skip_idx)
            skip_idx+=1
               
        layers += [nn.Flatten(),
                    nn.Linear((self.conv_output_resolution**2)*self.hidden_channels[-1], self.latent_dimension)]

        self.layers = nn.ModuleList(layers)
       
    
    def forward(self, x):

        for layer , layer_idx in zip(self.layers, range(len(self.layers))):
            x = layer(x)
            if layer_idx in self.skip_idxs:

                self.trace.append(x)
        self.trace.append(x)
        return  self.trace




class ConvBlock(nn.Module):
    """ 
    Standalone convolutional block. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, inputs):
        return self.block(inputs)

class DownConv(nn.Module):
    """
    Conv block used to downsample the height and width of an image.
    Also may adjusts the number of features if desired
    """
    def __init__(self, channel_in_dim, channel_out_dim, kernel_size, stride, padding, use_dropout:bool= True, skip_con: bool = True):# padding_type, norm_layer, use_dropout, use_bias,):
        super().__init__()
        
        self.in_dim = channel_in_dim
        self.out_dim = channel_out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_dropout = use_dropout
        self.skip_con = skip_con
        if skip_con:
            self.residual_pooler = nn.Sequential(*[nn.AvgPool2d(kernel_size =2, stride= 2, padding =0),
                                nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1)])
        self.act_fn =nn.LeakyReLU()                                
        self.conv_block = self.build_conv_block()

    def build_conv_block(self):
        conv_block = []
        conv_block += [
                    nn.Conv2d(self.in_dim, self.out_dim, self.kernel_size, 2, self.padding),
                    nn.BatchNorm2d(self.out_dim),
                    nn.LeakyReLU()]
        if self.use_dropout:
            conv_block += [nn.Dropout(0.3)]
        
        conv_block += [nn.Conv2d(self.out_dim, self.out_dim, self.kernel_size, self.stride, self.padding),
                        nn.BatchNorm2d(self.out_dim)]
        return nn.Sequential(*conv_block)

    def forward(self, inputs):
        if self.skip_con:
            return self.act_fn(self.residual_pooler(inputs)+ self.conv_block(inputs))

        return  self.act_fn(self.conv_block)
       


class ResnetBlock(nn.Module):
    """ 
    ResNet convolutional block.
    """
    def __init__(self, channel_dim, kernel_size, stride, padding, use_dropout):# padding_type, norm_layer, use_dropout, use_bias,):
        super().__init__()
        
        self.dim = channel_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_dropout = use_dropout
        self.act_fn = nn.LeakyReLU()
        self.conv_block = self.build_conv_block()

    def build_conv_block(self):
        conv_block = []
        conv_block += [
                    nn.Conv2d(self.dim, self.dim, self.kernel_size, self.stride, self.padding),
                    nn.BatchNorm2d(self.dim),
                    nn.LeakyReLU()]
        if self.use_dropout:
            conv_block += [nn.Dropout(0.3)]
        
        conv_block += [nn.Conv2d(self.dim, self.dim, self.kernel_size, self.stride, self.padding),
                        nn.BatchNorm2d(self.dim)]
  

        return nn.Sequential(*conv_block)

    def forward(self, inputs):
        return self.act_fn(inputs + self.conv_block(inputs))
       





if __name__ == '__main__':
    x = ContractingNet(3,256, [64,128,256,512], [4,4,4,4],16, 100)
    print(x)
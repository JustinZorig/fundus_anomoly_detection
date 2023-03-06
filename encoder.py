import torch
import torch.nn as nn




class EncoderNetwork(nn.Module):
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
        

        self.model = self._create_network()

    def _create_network(self):
    
        layers = [ConvBlock(self.input_channels, self.hidden_channels[0], 3, 1, 1)]

        for channels, repeat_val in zip(self.hidden_channels, self.layer_repetitions):

            for block_num in range(repeat_val-1):
                layers += [ResnetBlock(channels, 3, 1, 1, use_dropout = True)] 
            layers += [ResnetBlock(channels, 3, 1, 1, use_dropout = True, downsample = True)]            #we make the last resnet per block downsampke        
          #  layers += [nn.AvgPool2d(kernel_size = 2, stride = 2, padding =0)]  # downsamples image by factor of 2

        layers += [nn.Flatten(),
                    nn.Linear((self.conv_output_resolution**2)*self.hidden_channels[-1], self.latent_dimension)]

        self.net = nn.Sequential(*layers)
    
    def forward(self, inputs):
        return self.net(inputs)



class ConvBlock(nn.Module):
    """ 
    Standalone convolutional block. Primarily used at the beginning of the network 
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


class ResnetBlock(nn.Module):
    """ 
    ResNet convolutional block.
    """
    def __init__(self, channel_dim, kernel_size, stride, padding, use_dropout,downsample= False):# padding_type, norm_layer, use_dropout, use_bias,):
        super().__init__()
        
        self.dim = channel_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_dropout = use_dropout
        self.downsample = downsample
        self.downsampler= nn.AvgPool2d(kernel_size = 2, stride = 2, padding =0)
        self.conv_block = self.build_conv_block()

    def build_conv_block(self):
        conv_block = []
        conv_block += [
                    nn.Conv2d(self.dim, self.dim, self.kernel_size, self.stride, self.padding),
                    nn.BatchNorm2d(self.dim),
                    nn.LeakyReLU()]
        if self.use_dropout:
            conv_block += [nn.Dropout(0.3)]
        
        if self.downsample:
            conv_block += [nn.Conv2d(self.dim, self.dim, self.kernel_size, self.stride*2, self.padding),
                            nn.BatchNorm2d(self.dim)]
        else:
            conv_block += [nn.Conv2d(self.dim, self.dim, self.kernel_size, self.stride, self.padding),
                            nn.BatchNorm2d(self.dim)]
  

        return nn.Sequential(*conv_block)

    def forward(self, inputs):

        if self.downsample:
            out = self.downsampler(inputs) + self.conv_block(inputs)
        else :
            out = inputs + self.conv_block(inputs)
        return out






if __name__ == '__main__':
    x = EncoderNetwork(3,256, [64,128,256,512], [4,4,4,4],16, 100)
    print(x)
import torch
import torch.nn as nn




class DecoderNetwork(nn.Module):
    def __init__(self,
                output_channels,
                output_resolution,

                hidden_layer_channels: list,
                layer_repetitions: list,

                conv_input_resolution,
                latent_dimension ):
        """
        output_channel: number of channels of decoder output. Should match the input channels of the encoder input
        conv_input_resolution: integer representing the resolution that the hidden image takes on 
                                when input into the convolutional stage of the decoder. 
        """
        super().__init__()
        
        self.output_channels = output_channels
        self.output_resolution = output_resolution
        self.hidden_layer_channels = hidden_layer_channels
        self.layer_repetitions = layer_repetitions
        self.conv_input_resolution = conv_input_resolution
        self.latent_dimension = latent_dimension

        self.model = self._create_network()


    def _create_network(self):
        
        self.linear = nn.Linear(self.latent_dimension, (self.conv_input_resolution**2) * self.hidden_layer_channels[0])

        layers=[]
        for channels, repeat_val in zip(self.hidden_layer_channels, self.layer_repetitions):
            for block_num in range(repeat_val-1):
                layers += [ResnetBlock(channels, upsample =False)] 
            layers += [ResnetBlock(channels, upsample = True)]                 
           

        layers += [ConvBlock(self.hidden_layer_channels[-1], self.output_channels, 3, 1, 1)]


        self.net = nn.Sequential(*layers) 
    
    def forward(self, inputs):
        x = self.linear(inputs)
        x = x.reshape(x.shape[0], -1, self.conv_input_resolution, self.conv_input_resolution)
        return self.net(x)


class ConvBlock(nn.Module):
    """ 
    Standalone convolutional block. Primarily used at the beginning of the network 
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        #    nn.LeakyReLU(),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, inputs):
        return self.block(inputs)


class ResnetBlock(nn.Module):
    """ 
    ResNet convolutional block.
    """
    def __init__(self, channel_dim, upsample:bool, kernel_size=3, stride=1, padding=1,
                    up_stride=2, up_padding=1, up_out_padding=1, use_dropout=False):# padding_type, norm_layer, use_dropout, use_bias,):
        super().__init__()
        
        self.dim = channel_dim
        self.upsample = upsample


        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.up_stride = up_stride
        self.up_padding = up_padding
        self.up_out_padding = up_out_padding

        self.use_dropout = use_dropout
        self.conv_block = self.build_conv_block()
        self.upsampler = nn.Upsample(scale_factor = 2.0, mode = "nearest")

    def build_conv_block(self):
        conv_block = []
        if self.upsample:
            conv_block += [
                        nn.ConvTranspose2d(self.dim, self.dim, self.kernel_size, self.up_stride, self.up_padding, self.up_out_padding),
                        nn.BatchNorm2d(self.dim),
                        nn.LeakyReLU()]
        else:   
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
        if self.upsample:
            out = self.upsampler(inputs) + self.conv_block(inputs)
        else:
            out = inputs + self.conv_block(inputs)
        return out






if __name__ == '__main__':

    ww = [64,128,256,512]
    ww.reverse()
    print(ww) 
    print(type(ww))
    x = DecoderNetwork(3,  256, ww, [4,4,4,4], 16,100)
    print(x)
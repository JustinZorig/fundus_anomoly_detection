import torch
import torch.nn as nn




class ExpandingNet(nn.Module):
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
        self.hidden_channels = hidden_layer_channels
        self.layer_repetitions = layer_repetitions
        self.conv_input_resolution = conv_input_resolution
        self.latent_dimension = latent_dimension
    #    self.trace = []
        self.skip_idxs = []
        self.model = self._create_network()


    def _create_network(self):
        
        self.linear = nn.Linear(self.latent_dimension, (self.conv_input_resolution**2) * self.hidden_channels[0])

        layers=[]

        skip_idx =0

        for channels, repeat_val, idx in zip(self.hidden_channels, self.layer_repetitions, range(len(self.hidden_channels))):            
            for block_num in range(repeat_val):
                if block_num ==0  and idx!=0:  # No need to reduce channels for the first resnet blocks (skip con not used) 
                    layers += [ResnetBlock(channels, reduce_channels= True)]
                    self.skip_idxs.append(skip_idx)
                else:
                    layers += [ResnetBlock(channels)] 
                skip_idx+=1

            if channels!= self.hidden_channels[-1]:
                layers += [UpConv(channels, self.hidden_channels[idx+1],3,1,1)]      
            skip_idx+=1 
                      
           

        layers += [ConvBlock(self.hidden_channels[-1], self.output_channels, 3, 1, 1)]


        self.layers = nn.ModuleList(layers) 
    
    def forward(self, inputs): # inputs is in the order of the encoder part


        x = self.linear(inputs[-1])
        x = x.reshape(x.shape[0], -1, self.conv_input_resolution, self.conv_input_resolution)

        inputs.reverse()
        input_idx  =1

        for layer, layer_idx in zip(self.layers, range(len(self.layers))):
            if layer_idx in self.skip_idxs:
                x= torch.cat((x, inputs[input_idx]), axis =1)
                input_idx+=1
            x = layer(x)
        del inputs            
        return x


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

class UpConv(nn.Module):
    """
    Conv block used to upsamplethe height and width of an image.
    Also may adjusts the number of features if desired
    """
    def __init__(self, channel_in_dim, channel_out_dim, kernel_size, stride, padding, use_dropout:bool= True, skip_con: bool = True):# padding_type, norm_layer, use_dropout, use_bias,):
        super().__init__()
        
        self.in_dim = channel_in_dim
        self.out_dim = channel_out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding =1
        self.out_padding =1
        self.use_dropout = use_dropout
        self.skip_con = skip_con
        self.act_fn = nn.LeakyReLU()
        if skip_con:
            self.residual = nn.Sequential(*[nn.Upsample(scale_factor = self.stride*2, mode = "nearest"),
                                nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1)])
        self.conv_block = self.build_conv_block()

    def build_conv_block(self):
        conv_block = []
        conv_block += [
                    nn.ConvTranspose2d(self.in_dim, self.out_dim, self.kernel_size, self.stride*2, self.padding, self.out_padding),
                        nn.BatchNorm2d(self.out_dim),
                        nn.LeakyReLU()]
        if self.use_dropout:
            conv_block += [nn.Dropout(0.3)]
        
        conv_block += [nn.Conv2d(self.out_dim, self.out_dim, self.kernel_size, self.stride, self.padding),
                        nn.BatchNorm2d(self.out_dim)]
        return nn.Sequential(*conv_block)

    def forward(self, inputs):
        if self.skip_con:
            return self.act_fn(self.residual(inputs)+ self.conv_block(inputs))

        return  self.act_fn(self.conv_block)

class ResnetBlock(nn.Module):
    """ 
    ResNet convolutional block.
    """
    def __init__(self, channel_dim, kernel_size=3, stride=1, padding=1, use_dropout:bool = True, reduce_channels:bool = False):# padding_type, norm_layer, use_dropout, use_bias,):
        super().__init__()
        
        self.dim = channel_dim
        self.in_dim = channel_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_dropout = use_dropout
        self.reduce_channels = reduce_channels
        if reduce_channels: 
            self.in_dim =self.dim*2
            self.residual = nn.Conv2d(self.in_dim, self.dim, kernel_size =1, stride =1, padding=0 )

        self.act_fn=nn.LeakyReLU()
        self.conv_block = self.build_conv_block()

    def build_conv_block(self):
 
        conv_block = []
        conv_block += [
                    nn.Conv2d(self.in_dim, self.dim, self.kernel_size, self.stride, self.padding),
                    nn.BatchNorm2d(self.dim),
                    nn.LeakyReLU()]
        if self.use_dropout:
            conv_block += [nn.Dropout(0.3)]
        conv_block += [nn.Conv2d(self.dim, self.dim, self.kernel_size, self.stride, self.padding),
                        nn.BatchNorm2d(self.dim)]
  

        return nn.Sequential(*conv_block)

    def forward(self, inputs):
        if self.reduce_channels:
            return self.act_fn(self.residual(inputs) + self.conv_block(inputs))
        return self.act_fn(inputs + self.conv_block(inputs))
       
       



if __name__ == '__main__':

    ww = [64,128,256,512]
    ww.reverse()
    print(ww) 
    print(type(ww))
    x = ExpandingNet(3,  256, ww, [4,4,4,4], 16,100)
    print(x)
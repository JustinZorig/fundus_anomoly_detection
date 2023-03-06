import torch
import torch.nn as nn




class ExpandingNetSimple(nn.Module):
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

        self.trace = []

        self._create_network()


    def _create_network(self):
        
        self.linear = nn.Linear(self.latent_dimension, (self.conv_input_resolution**2) * self.hidden_channels[0])

        layers=[]


        for channels, repeat_val, idx in zip(self.hidden_channels, self.layer_repetitions, range(len(self.hidden_channels))):            
            for block_num in range(repeat_val):
                if block_num ==0  and idx!=0:  # No need to reduce channels for the first resnet blocks (skip con not used) 
                    layers += [ConvBlock(self.hidden_channels[idx-1], channels, 3,1,1)]
                else: 
                    layers += [ConvBlock(channels, channels, 3,1,1)]
               

            if channels!= self.hidden_channels[-1]:
                layers += [UpConv(channels, self.hidden_channels[idx+1],3,stride=2,padding =1)]      
                
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
    Standalone convolutional block. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_dropout = True):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_dropout = use_dropout
        self._build_block()


    def _build_block(self):
        conv_block = []
        conv_block +=[
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.out_channels)]
        
        if self.use_dropout:
            conv_block += [nn.Dropout(0.3)]
        
        self.block = nn.Sequential(*conv_block)
    
    def forward(self, inputs):
        return self.block(inputs)

class UpConv(nn.Module):
    """
    Conv block used to upsamplethe height and width of an image.
    Also may adjusts the number of features if desired
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_dropout:bool= True):# padding_type, norm_layer, use_dropout, use_bias,):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding =1
        self.out_padding =1
        self.use_dropout = use_dropout
        
        self._build_block()

    def _build_block(self):
        conv_block = []
        conv_block += [
                    nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.out_padding),
                        nn.BatchNorm2d(self.out_channels),
                        nn.LeakyReLU()]
        if self.use_dropout:
            conv_block += [nn.Dropout(0.3)]
        
  
        self.block = nn.Sequential(*conv_block)

    def forward(self, inputs):
        return  self.conv_block




if __name__ == '__main__':

    ww = [64,128,256,512]
    ww.reverse()
    print(ww) 
    print(type(ww))
    x = ExpandingNetSimple(3,  256, ww, [2,2,2,2], 16,100)
    print(x)
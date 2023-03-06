import torch
import torch.nn as nn




class ContractingNetSimple(nn.Module):
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
        self.skip_idxs=[]
        self._create_network()

    def _create_network(self):
    
        layers = [ConvBlock(self.input_channels, self.hidden_channels[0], 3, 1, 1, use_dropout=False)]

        skip_idx=0
        for channels, repeat_val, idx in zip(self.hidden_channels, self.layer_repetitions, range(len(self.hidden_channels))):
            for block_num in range(repeat_val):
                layers += [ConvBlock(channels,channels, 3, 1, 1, use_dropout = True)]
                skip_idx+=1
        
            if channels != self.hidden_channels[-1]:
                layers += [DownConv(channels,self.hidden_channels[idx+1], 3, stride =2, padding=1, use_dropout = True)]
            else:
                layers += [DownConv(channels, channels*2, 3, stride =2, padding=1, use_dropout = True)]
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

class DownConv(nn.Module):
    """
    Conv block used to downsample the height and width of an image.
    Also may adjusts the number of features if desired
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_dropout:bool= True):# padding_type, norm_layer, use_dropout, use_bias,):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_dropout = use_dropout   

        self._build_block()

    def _build_block(self):
        conv_block = []
        conv_block += [
                    nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
                    nn.BatchNorm2d(self.out_channels),
                    nn.LeakyReLU()]
        if self.use_dropout:
            conv_block += [nn.Dropout(0.3)]
       
        self.block = nn.Sequential(*conv_block)

    def forward(self, inputs):
       
        return self.block(inputs)
    


if __name__ == '__main__':
    x = ContractingNetSimple(3,256, [64,128,256,512], [2,2,2,2],16, 100)
    #y= ContractingNetSimple(input_channels, input_resolution, hidden_layer_channels, layer_repetitions, conv_output_resolution, latent_dimension)
    print(x)
# This file was modified from https://github.com/PRBonn/lidar-bonnetal

import torch
from collections import OrderedDict

class ConvBlock(torch.nn.Module):
    def __init__(self, channels, mid_channels, momentum, slope):
        super(ConvBlock, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=channels, out_channels=mid_channels, 
                                    kernel_size=(1, 1), stride=1, 
                                    padding=0, bias=False)
        self.bn_1 = torch.nn.BatchNorm2d(num_features=mid_channels, momentum=momentum)
        self.relu_1 = torch.nn.LeakyReLU(negative_slope=slope)
        
        self.conv_2 = torch.nn.Conv2d(in_channels=mid_channels, out_channels=channels, 
                                     kernel_size=(3, 3), stride=1, 
                                     padding=1, bias=False)
        self.bn_2 = torch.nn.BatchNorm2d(num_features=channels, momentum=momentum)
        self.relu_2 = torch.nn.LeakyReLU(negative_slope=slope)
    
    def forward(self, x):
        residual = x # id assignment
        
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu_1(out)
        
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu_2(out)
        
        out = out + residual
        return out
        
        
class DarknetEncoder(torch.nn.Module):
    model_blocks = {
    21: [1, 1, 2, 2, 1],
    53: [1, 2, 8, 8, 4],
    }
    def __init__(self, layers_number, in_channels, momentum, slope, dropout_p):
        super(DarknetEncoder, self).__init__()
        if layers_number not in DarknetEncoder.model_blocks.keys():
            raise TypeError(f"layers_number MUST be in {DarknetEncoder.model_blocks.keys}")
        layers_list = DarknetEncoder.model_blocks[layers_number]
        self.in_channels = in_channels
        self.momentum = momentum
        self.slope = slope
        self.dropout_p = dropout_p
        self.os_dict = dict()
        
        self.conv_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=32, 
                                      kernel_size=(3, 3), stride=1,
                                      padding=1, bias=False)
        self.bn_1 = torch.nn.BatchNorm2d(32, momentum=self.momentum)
        self.relu_1 = torch.nn.LeakyReLU(negative_slope=self.slope)
        
        self.encoder_1 = self.__make_encoder_layer(in_channels=32, out_channels=64, 
                                                   blocks_number=layers_list[0])
        self.encoder_2 = self.__make_encoder_layer(in_channels=64, out_channels=128, 
                                                   blocks_number=layers_list[0])
        self.encoder_3 = self.__make_encoder_layer(in_channels=128, out_channels=256, 
                                                   blocks_number=layers_list[0])
        self.encoder_4 = self.__make_encoder_layer(in_channels=256, out_channels=512, 
                                                   blocks_number=layers_list[0])
        self.encoder_5 = self.__make_encoder_layer(in_channels=512, out_channels=1024, 
                                                   blocks_number=layers_list[0])
    
        self.dropout = torch.nn.Dropout2d(p=self.dropout_p)
    
    def __make_encoder_layer(self, in_channels, out_channels, blocks_number):
        layers = []
        
        layers.append(('conv', torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=(3, 3), stride=(1, 2), 
                                               padding=1, bias=False)))
        layers.append(('bn', torch.nn.BatchNorm2d(num_features=out_channels, momentum=self.momentum)))
        layers.append(('relu', torch.nn.LeakyReLU(negative_slope=self.slope)))
        
        for i in range(blocks_number):
            layers.append((f'conv_block_{i}', 
                           ConvBlock(channels=out_channels, mid_channels=in_channels, 
                                     momentum=self.momentum, slope=self.slope)))
        
        return torch.nn.Sequential(OrderedDict(layers))
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        self.os_dict['os1'] = x.detach()
        
        x = self.encoder_1(x)
        self.os_dict['os2'] = x.detach()
        x = self.dropout(x)
        
        x = self.encoder_2(x)
        self.os_dict['os4'] = x.detach()
        x = self.dropout(x)
        
        x = self.encoder_3(x)
        self.os_dict['os8'] = x.detach()
        x = self.dropout(x)
        
        x = self.encoder_4(x)
        self.os_dict['os16'] = x.detach()
        x = self.dropout(x)
        
        x = self.encoder_5(x)
        self.os_dict['os32'] = x.detach()
        return x, self.os_dict
    
class DarknetDecoder(torch.nn.Module):
    def __init__(self, out_channels, momentum, slope, dropout_p):
        super(DarknetDecoder, self).__init__() 
        self.out_channels = out_channels
        self.momentum = momentum
        self.slope = slope
        self.os_dict = None
        self.dropout_p = dropout_p
        
        self.decoder_1 = self.__make_decoder_layer(in_channels=1024, out_channels=512)
        self.decoder_2 = self.__make_decoder_layer(in_channels=512, out_channels=256)
        self.decoder_3 = self.__make_decoder_layer(in_channels=256, out_channels=128)
        self.decoder_4 = self.__make_decoder_layer(in_channels=128, out_channels=64)
        self.decoder_5 = self.__make_decoder_layer(in_channels=64, out_channels=32)
        self.dropout = torch.nn.Dropout(p=self.dropout_p)
        self.conv = torch.nn.Conv2d(in_channels=32, out_channels=self.out_channels,
                                    kernel_size=(3, 3), stride=1, padding=1) 
    
    def __make_decoder_layer(self, in_channels, out_channels):
        layers = []
        
        layers.append(('upconv', torch.nn.ConvTranspose2d(in_channels=in_channels, 
                                                          out_channels=out_channels,
                                                          kernel_size=(1, 4), stride=(1, 2), 
                                                          padding=(0, 1), bias=False)))
        layers.append(('bn', torch.nn.BatchNorm2d(num_features=out_channels, momentum=self.momentum)))
        layers.append(('relu', torch.nn.LeakyReLU(negative_slope=self.slope)))
        
        layers.append(('conv_block', ConvBlock(channels=out_channels, mid_channels=in_channels, 
                                               momentum=self.momentum, slope=self.slope)))
        
        return torch.nn.Sequential(OrderedDict(layers))

    def forward(self, x, os_dict):
        x = self.decoder_1(x)
        x = x + os_dict['os16']
        x = self.decoder_2(x)
        x = x + os_dict['os8']
        x = self.decoder_3(x)
        x = x + os_dict['os4']
        x = self.decoder_4(x)
        x = x + os_dict['os2']
        x = self.decoder_5(x)
        x = x + os_dict['os1']
        x = self.dropout(x)
        x = self.conv(x)
        return x
    
    
class Darknet(torch.nn.Module):
    def __init__(self, layers_number, in_channels, out_channels, momentum, slope, dropout_p):
        super(Darknet, self).__init__()
        self.encoder = DarknetEncoder(layers_number=layers_number, in_channels=in_channels, 
                                      momentum=momentum, slope=slope, dropout_p=dropout_p)
        self.decoder = DarknetDecoder(out_channels=out_channels, momentum=momentum, 
                                      slope=slope, dropout_p=dropout_p) 
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x, os_dict = self.encoder(x)
        x = self.decoder(x, os_dict)
        return x







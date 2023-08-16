import torch

from backbone.Darknet import Darknet
from utils.RangeImage import RangeImage

import time

if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    range_image, semantic_image, mapping_v, mapping_u = \
        RangeImage.mapping("C:/Users/1015947658/Desktop/rangenet-pytorch/src/example/000000.bin",
                           "C:/Users/1015947658/Desktop/rangenet-pytorch/src/example/000000.label")
    
    backbone = Darknet(layers_number=21, in_channels=5, out_channels=8, 
                       momentum=0.01, slope=0.1, dropout_p=0.01)
    backbone.to(device=device)
    backbone.eval()
    with torch.no_grad():
        range_image = torch.from_numpy(range_image).float()
        range_image = torch.unsqueeze(range_image, dim=0)
        range_image = range_image.to(device=device)
        
        t0 = time.time()
        range_image_ = backbone(range_image)
        print(time.time()-t0)
        
        print(range_image_[0, :, 45, 488])
        # print(range_image_)
    
   
    
    
    
import torch
import yaml
import time

from backbone.Darknet import Darknet
from utils.RangeImage import RangeImage
from utils.SemDataset import SemTrainDataset

if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    range_image, semantic_image, mapping_v, mapping_u = \
        RangeImage.mapping("C:/Users/1015947658/Desktop/rangenet-pytorch/src/example/000000.bin",
                           "C:/Users/1015947658/Desktop/rangenet-pytorch/src/example/000000.label")
    
    backbone = Darknet(layers_number=21, in_channels=5, out_channels=20, 
                       momentum=0.01, slope=0.1, dropout_p=0.01)
    backbone.to(device=device)
    backbone.train()
    range_image = torch.from_numpy(range_image).float()
    range_image = torch.unsqueeze(range_image, dim=0)
    range_image = range_image.to(device=device)
    
    dataset_cfg_path = "C:/Users/1015947658/Desktop/github/rangenet-pytorch/src/config/semantic-kitti.yaml"
    with open(dataset_cfg_path, 'r') as dataset_cfg_file:
        dataset_cfg = yaml.safe_load(dataset_cfg_file)
    learning_map = dataset_cfg['learning_map']
    for key, value in learning_map.items():
        semantic_image[semantic_image==key] = value
    
    semantic_image = torch.from_numpy(semantic_image).float()
    semantic_image = torch.unsqueeze(semantic_image, dim=0)
    semantic_image = semantic_image.to(device=device)
    
    t0 = time.time()
    range_image = backbone(range_image)
    print(time.time()-t0)
    
    criterion = torch.nn.NLLLoss(ignore_index=-1)
    criterion.to(device=device)
    
    range_image = torch.log(range_image)
    
    print(range_image.shape,semantic_image.shape)
    
    loss = criterion(range_image, semantic_image)
    # loss.backward()
    print(loss.item())

    dataset = SemDataset(dataset_dir=r"C:/Users/1015947658/Desktop/semanticKITTI",
                         split='test', 
                         dataset_cfg_path=r"C:/Users/1015947658/Desktop/github/rangenet-pytorch/src/config/semantic-kitti.yaml")
    range_image, mapping_v, mapping_u = dataset[3]
    print(range_image.shape, semantic_image.shape)
    
    
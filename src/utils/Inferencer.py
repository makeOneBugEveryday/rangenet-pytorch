
import yaml
import torch

from utils.SemDataset import SemSequenceDataset
from backbone.Darknet import Darknet
from postprocess.Reconstruction import Reconstruction

class Inferencer:
    def __init__(self, inferencer_cfg_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(inferencer_cfg_path, 'r') as inferencer_cfg_file:
            inferencer_cfg = yaml.safe_load(inferencer_cfg_file)
        self.pcd_dir = inferencer_cfg['inferencer']['pcd_dir']
        self.dataset_cfg_path = inferencer_cfg['inferencer']['dataset_cfg_path']
        self.label_dir = inferencer_cfg['inferencer']['label_dir']
        self.layers_number = inferencer_cfg['darknet']['layers_number']
        self.in_channels = inferencer_cfg['darknet']['in_channels']
        self.out_channels = inferencer_cfg['darknet']['out_channels']
        self.momentum = inferencer_cfg['darknet']['momentum']
        self.slope = inferencer_cfg['darknet']['slope']
        self.dropout_p = inferencer_cfg['darknet']['dropout_p']
        
        self.dataset = SemSequenceDataset(pcd_dir=self.pcd_dir,
                                          dataset_cfg_path=self.dataset_cfg_path,
                                          label_dir=self.label_dir)
        
        self.backbone = Darknet(layers_number=self.layers_number,
                                in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                momentum=self.momentum,
                                slope=self.slope,
                                dropout_p=self.dropout_p)
        self.backbone.to(device=self.device)
        self.backbone.eval()
        
        self.postprocess = Reconstruction(use_knn=False)
        
    def infer_single(self, index):
        range_image, semantic_image, mapping_v, mapping_u = self.dataset[index]
        range_image = torch.unsqueeze(range_image, dim=0)
        range_image = range_image.to(device=self.device)
        prediction = self.backbone(range_image)
        prediction = torch.argmax(prediction, dim=1)
        prediction = torch.squeeze(prediction, dim=0).cpu().detach().numpy()
        
        pcd_prediction = self.postprocess.mapping(prediction=prediction, 
                               mapping_v=mapping_v, mapping_u=mapping_u)
        print(pcd_prediction)
    
    def infer_sence(self):
        for range_image, semantic_image, mapping_v, mapping_u in self.dataset:
            pass

















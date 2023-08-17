
import yaml
import torch
from torch.utils.data import DataLoader

from utils.SemDataset import SemTrainDataset
from backbone.Darknet import Darknet

class Trainer:
    def __init__(self, trainer_cfg_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        with open(trainer_cfg_path, 'r') as trainer_cfg_file:
            trainer_cfg = yaml.safe_load(trainer_cfg_file)
        dataset_dir = trainer_cfg['trainer']['dataset_dir']
        dataset_cfg_path = trainer_cfg['trainer']['dataset_cfg_path']
        batch_size = trainer_cfg['trainer']['batch_size']
        shuffle = trainer_cfg['trainer']['shuffle']
        num_workers = trainer_cfg['trainer']['num_workers']
        lr = trainer_cfg['trainer']['lr']
        self.max_epochs = trainer_cfg['trainer']['max_epochs']
        
        layers_number = trainer_cfg['darknet']['layers_number']
        in_channels = trainer_cfg['darknet']['in_channels']
        out_channels = trainer_cfg['darknet']['out_channels']
        momentum = trainer_cfg['darknet']['momentum']
        slope = trainer_cfg['darknet']['slope']
        dropout_p = trainer_cfg['darknet']['dropout_p']
        
        self.dataset = SemTrainDataset(dataset_dir=dataset_dir,
                                  dataset_cfg_path=dataset_cfg_path)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size,
                                     shuffle=shuffle, num_workers=num_workers)
        
        self.backbone = Darknet(layers_number=layers_number, in_channels=in_channels, 
                                out_channels=out_channels, momentum=momentum, 
                                slope=slope, dropout_p=dropout_p)
        self.backbone.train()
        self.backbone.to(device=self.device)
        
        weights = self.dataset.get_weights()
        self.criterion = torch.nn.NLLLoss(weight=weights, ignore_index=-1)
        self.criterion.to(device=self.device)
        
        self.optimizer = torch.optim.SGD(self.backbone.parameters(), lr=lr)
        
    
    def train(self, epochs=None):
        if epochs is None: 
            epochs = self.max_epochs
        for epoch in range(epochs):
            for i, (range_image, semantic_image) in enumerate(self.dataloader):
                print(i, range_image.shape, semantic_image.shape)
                range_image = range_image.to(device=self.device)
                semantic_image = semantic_image.to(device=self.device)
                
                output = self.backbone(range_image)
                output = torch.log(output)
                print(output.shape, semantic_image.shape)
                loss = self.criterion(output, semantic_image)
                print(loss.item())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

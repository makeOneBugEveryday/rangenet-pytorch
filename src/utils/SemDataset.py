
import torch
import yaml
import os
from torch.utils.data import Dataset

from utils.RangeImage import RangeImage

class SemTrainDataset(Dataset):
    def __init__(self, dataset_dir, dataset_cfg_path):
        self.dataset_dir = dataset_dir
        self.split = 'train'

        with open(dataset_cfg_path, 'r') as dataset_cfg_file:
            dataset_cfg = yaml.safe_load(dataset_cfg_file)
        self.labels = dataset_cfg['labels']
        self.color_map = dataset_cfg['color_map']
        self.content = dataset_cfg['content']
        self.learning_map = dataset_cfg['learning_map']
        self.learning_map_inv = dataset_cfg['learning_map_inv']
        self.learning_ignore = dataset_cfg['learning_ignore']
        self.sequence_list = dataset_cfg['split'][self.split]
        
        sequence_list = ["{:0>2d}".format(int(item)) for item in self.sequence_list]
        pcd_dir_list = [os.path.join(self.dataset_dir, 'sequences', item, 'velodyne') 
                        for item in sequence_list]
        self.pcd_list = []
        for pcd_dir_item in pcd_dir_list:
            self.pcd_list.extend([os.path.join(pcd_dir_item, item) 
                                  for item in os.listdir(pcd_dir_item)])
            
        label_dir_list = [os.path.join(self.dataset_dir, 'sequences', item, 'labels') 
                          for item in sequence_list]
        self.label_list = []
        for label_item in label_dir_list:
            self.label_list.extend([os.path.join(label_item, item) 
                                    for item in os.listdir(label_item)])
                
    def __getitem__(self, index):
        pcd_path = self.pcd_list[index]
        label_path = self.label_list[index]
        range_image, semantic_image, mapping_v, mapping_u = \
            RangeImage.mapping(pcd_path=pcd_path, label_path=label_path)
        for key, value in self.learning_map.items():
            semantic_image[semantic_image==key] = value
        return torch.from_numpy(range_image).float(), torch.from_numpy(semantic_image).long()
        
    def __len__(self):
        return len(self.pcd_list)
    
    def get_weights(self):
        weights = dict()
        for (key, value) in self.content.items():
            if self.learning_map[key] not in weights.keys():
                weights[self.learning_map[key]] = value
            else:
                weights[self.learning_map[key]] += value
            if self.learning_ignore[self.learning_map[key]]:
                weights[self.learning_map[key]] = 0
        return torch.Tensor(list(weights.values()))
    

class SemSequenceDataset(Dataset):
    def __init__(self, pcd_dir, dataset_cfg_path, *, label_dir=None):
        self.pcd_dir = pcd_dir
        self.label_dir = None
        if label_dir is not None:
            self.label_dir = label_dir
        
        with open(dataset_cfg_path, 'r') as dataset_cfg_file:
            dataset_cfg = yaml.safe_load(dataset_cfg_file)
        self.labels = dataset_cfg['labels']
        self.color_map = dataset_cfg['color_map']
        self.content = dataset_cfg['content']
        self.learning_map = dataset_cfg['learning_map']
        self.learning_map_inv = dataset_cfg['learning_map_inv']
        self.learning_ignore = dataset_cfg['learning_ignore']

        self.pcd_list = [os.path.join(pcd_dir, item) 
                              for item in os.listdir(pcd_dir)]
        if label_dir is not None:
            self.label_list = [os.path.join(label_dir, item) 
                              for item in os.listdir(label_dir)]

    def __getitem__(self, index):
        pcd_path = self.pcd_list[index]
        label_path = None
        if self.label_dir is not None:
            label_path = self.label_list[index]
        range_image, semantic_image, mapping_v, mapping_u = \
            RangeImage.mapping(pcd_path=pcd_path, label_path=label_path)
            
        range_image = torch.from_numpy(range_image).float()
        if semantic_image is not None:
            for key, value in self.learning_map.items():
                semantic_image[semantic_image==key] = value
        return range_image, semantic_image, mapping_v, mapping_u
        
    def __len__(self):
        return len(self.sequence_list)
        







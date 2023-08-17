
import numpy as np

class Scanloader:
    pcd_extension = ['.bin']
    label_extension = ['.label']
    
    @staticmethod
    def load_pcd(pcd_path):
        if pcd_path.endswith('.bin'):
            coordinate, remission = Scanloader.__load_pcd_bin(pcd_path)
        else: 
            raise TypeError(f"point cloud data should end with in {Scanloader.pcd_extension}")
        return coordinate, remission
    
    @staticmethod
    def load_label(label_path):
        if label_path.endswith('.label'):
            semantic_label = Scanloader.__load_label_lable(label_path)
        else: 
            raise TypeError(f"label data should end with in {Scanloader.label_extension}")
        return semantic_label
    
    @staticmethod
    def __load_pcd_bin(pcd_path):
        points = np.fromfile(pcd_path, dtype=np.float32)
        points = points.reshape((-1, 4))
        coordinate = points[:, 0:3]
        remission = points[:, 3]
        return coordinate, remission

    @staticmethod
    def __load_label_lable(label_path):
        label = np.fromfile(label_path, dtype=np.int32)
        label = label.reshape((-1))
        semantic_label = label & 0x0000FFFF
        # instance_lable = label >> 16
        return semantic_label
    
    











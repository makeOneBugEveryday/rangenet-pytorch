
import numpy as np
import time

class RangeImage:
    pcd_extension = ['.bin']
    label_extension = ['.label']
    fov_up, fov_down = 3, -25 # (3, -25) (2, -24.8)
    width, height = 1024, 64
    
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
        
    @staticmethod
    def mapping(pcd_path, label_path):
        if pcd_path.endswith('.bin'):
            coordinate, remission = RangeImage.__load_pcd_bin(pcd_path)
        else: 
            raise TypeError(f"point cloud data should end with in {RangeImage.pcd_extension}")
        
        if label_path.endswith('.label'):
            semantic_label = RangeImage.__load_label_lable(label_path)
        else: 
            raise TypeError(f"label data should end with in {RangeImage.label_extension}")
        
        (coordinate_x, coordinate_y, coordinate_z) = coordinate.T
        
        mapping_u = 0.5*(1-np.arctan2(coordinate_y, coordinate_x)/np.pi)
        
        fov_up = RangeImage.fov_up / 180.0 * np.pi
        fov_down = RangeImage.fov_down / 180.0 * np.pi
        fov = fov_up - fov_down
        # coordinate_r = np.sqrt(np.power(coordinate_x,2)+np.power(coordinate_y,2)+np.power(coordinate_z,2))
        coordinate_r = np.linalg.norm(coordinate, 2, axis=1)
        mapping_v = (1-(np.arcsin(coordinate_z/coordinate_r)-fov_down)/fov)
        mapping_v = (mapping_v-np.min(mapping_v))/(np.max(mapping_v)-np.min(mapping_v))*0.999
        
        mapping_u = np.floor(mapping_u*RangeImage.width).astype(int)
        mapping_v = np.floor(mapping_v*RangeImage.height).astype(int)
        
        range_image = np.full((5, RangeImage.height, RangeImage.width), 0, dtype=float)        
        range_image[0, mapping_v, mapping_u] = coordinate_x
        range_image[1, mapping_v, mapping_u] = coordinate_y
        range_image[2, mapping_v, mapping_u] = coordinate_z
        range_image[3, mapping_v, mapping_u] = coordinate_r
        range_image[4, mapping_v, mapping_u] = remission
        
        semantic_image = np.full((RangeImage.height, RangeImage.width), -1, dtype=int)
        semantic_image[mapping_v, mapping_u] = semantic_label
        
        return range_image, semantic_image, mapping_v, mapping_u
    
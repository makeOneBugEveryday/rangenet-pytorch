
import numpy as np

from utils.Scanloader import Scanloader

class RangeImage:
    fov_up, fov_down = 3, -25 # (3, -25) (2, -24.8)
    width, height = 1024, 64
        
    @staticmethod
    def mapping(pcd_path, label_path=None):

        coordinate, remission = Scanloader.load_pcd(pcd_path)
        (coordinate_x, coordinate_y, coordinate_z) = coordinate.T
        
        mapping_u = 0.5*(1-np.arctan2(coordinate_y, coordinate_x)/np.pi)
        
        fov_up = RangeImage.fov_up / 180.0 * np.pi
        fov_down = RangeImage.fov_down / 180.0 * np.pi
        fov = fov_up - fov_down
        # coordinate_r = np.sqrt(np.power(coordinate_x,2)+np.power(coordinate_y,2)+np.power(coordinate_z,2))
        coordinate_r = np.linalg.norm(coordinate, 2, axis=1)
        mapping_v = (1-(np.arcsin(coordinate_z/coordinate_r)-fov_down)/fov)
        mapping_v = (mapping_v-np.min(mapping_v))/(np.max(mapping_v)-np.min(mapping_v))
        
        mapping_u = np.floor(mapping_u*0.999*RangeImage.width).astype(int)
        mapping_v = np.floor(mapping_v*0.999*RangeImage.height).astype(int)
        
        range_image = np.full((5, RangeImage.height, RangeImage.width), 0, dtype=float)        
        range_image[0, mapping_v, mapping_u] = coordinate_x
        range_image[1, mapping_v, mapping_u] = coordinate_y
        range_image[2, mapping_v, mapping_u] = coordinate_z
        range_image[3, mapping_v, mapping_u] = coordinate_r
        range_image[4, mapping_v, mapping_u] = remission
        
        if label_path is None:
            semantic_image = None
        else:
            semantic_label = Scanloader.load_label(label_path)
            semantic_image = np.full((RangeImage.height, RangeImage.width), -1, dtype=int)
            semantic_image[mapping_v, mapping_u] = semantic_label
            
        return range_image, semantic_image, mapping_v, mapping_u
    
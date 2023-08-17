
import numpy as np

class Reconstruction:
    def __init__(self, use_knn=False):
        self.use_knn = use_knn
        if self.use_knn:
            pass

    def mapping(self, prediction, mapping_v, mapping_u):
        pcd_prediction = prediction[mapping_v, mapping_u]
        if self.use_knn:
            pass
        return pcd_prediction

        
    
    
    
    
    
    
    
    
    
    
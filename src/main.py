
from utils.SemDataset import SemTrainDataset, SemSequenceDataset

if __name__ == '__main__':
    dataset = SemSequenceDataset(pcd_dir="C:/Users/1015947658/Desktop/semanticKITTI/sequences/08/velodyne",
                                 dataset_cfg_path="C:/Users/1015947658/Desktop/github/rangenet-pytorch/src/config/semantic-kitti.yaml",
                                 label_dir="C:/Users/1015947658/Desktop/semanticKITTI/sequences/08/labels")

    range_image, semantic_image, mapping_v, mapping_u = dataset[0]
    print(range_image.shape, semantic_image, \
        mapping_v.shape, mapping_u.shape)















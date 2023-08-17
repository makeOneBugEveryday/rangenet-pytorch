from utils.Inferencer import Inferencer

if __name__ == '__main__':
    inferencer_cfg_path = "C:/Users/1015947658/Desktop/github/rangenet-pytorch/src/config/inferencer.yaml"
    
    inferencer = Inferencer(inferencer_cfg_path=inferencer_cfg_path)

    inferencer.infer_single(index=0)





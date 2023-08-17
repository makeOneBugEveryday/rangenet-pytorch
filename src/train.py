
from utils.Trainer import Trainer

if __name__ == '__main__':
    trainer_cfg_path = r"C:/Users/1015947658/Desktop/github/rangenet-pytorch/src/config/trainer.yaml"
    trainer = Trainer(trainer_cfg_path=trainer_cfg_path)

    trainer.train()




























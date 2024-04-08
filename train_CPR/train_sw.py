import hydra
from dataloader import sw_dataloader as dataloader
from model.SWnet import SWNET
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    train_dataloader, test_dataloader = dataloader.get_dataloaders(cfg)
        
if __name__ == "__main__":
    main()
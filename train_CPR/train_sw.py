import hydra
from dataloader import sw_dataloader as dataloader
from model.SWnet import SWNET
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(conf):
    if conf.smartwatch.get_stats:
        dataloader.get_data_stats(conf)
        return 1
    
    train_dataloader, test_dataloader = dataloader.get_dataloaders(conf)
        
if __name__ == "__main__":
    main()
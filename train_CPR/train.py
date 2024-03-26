import hydra
from dataloader import dataloader

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # Your code here
    dataset = dataloader.get_dataloaders(cfg)
    print(cfg)

if __name__ == "__main__":
    main()
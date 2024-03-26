import hydra
from dataloader import dataloader
from model.CPRnet import CPRnet

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    cpr_model=CPRnet(cfg)
    train_dataloader, test_dataloader = dataloader.get_dataloaders(cfg)
    for batch in train_dataloader:
        depth_img_ar,mask_img_ar,GT_depth,ts=batch
        cpr_model(depth_img_ar)
        break
    print(cfg)

if __name__ == "__main__":
    main()
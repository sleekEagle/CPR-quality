import hydra
from dataloader import dataloader
from model.CPRnet import CPRnet
import torch.nn as nn
import torch

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    loss_function = nn.MSELoss()
    cpr_model=CPRnet(cfg)
    optimizer = torch.optim.Adam(cpr_model.parameters(), lr=cfg.lr)
    train_dataloader, test_dataloader = dataloader.get_dataloaders(cfg)
    for batch in train_dataloader:
        optimizer.zero_grad()
        if cfg.type=='image':
            depth_img_ar,mask_img_ar,GT_depth,ts=batch
            cpr_model(depth_img_ar)
        elif cfg.type=='XYZ':
            XYZ_ar,GT_depth,ts=batch
            depth_pred=cpr_model(XYZ_ar)
            loss=loss_function(depth_pred,GT_depth.squeeze())
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss}")           
if __name__ == "__main__":
    main()
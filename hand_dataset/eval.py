import torch
import hydra
from omegaconf import DictConfig
import dataloader.dataloader as dataloader
from models.Handblur import AENet
import os

torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

def eval_depth(model,test_dataloader,device,conf):
    total_error=0
    gt_dists,rmse_errors=[],[]
    for i, batch in enumerate(test_dataloader):
        print(f"evaluating {i} of {len(test_dataloader)}",end='\r')
        img,depth,blur,seg=batch 
        img,depth,blur,seg=img.to(device),depth.to(device),blur.to(device),seg.to(device)
        img=img.swapaxes(2,3).swapaxes(1,2)
        with torch.no_grad():
            model.eval()
            pred_depth, pred_blur,_=model(img)

            mask=(seg>0) & (depth>0)
            pred_depth_actual=pred_depth*conf.max_depth
            GT_depth_actual=depth*conf.max_depth
            rmse_error=torch.square(torch.mean((pred_depth_actual.squeeze(dim=1)[mask]-GT_depth_actual[mask])**2))
            total_error+=rmse_error.item()
    acc=total_error/len(test_dataloader)
    print(f'Eval error: {acc*1000:.2f} mm')
    return acc

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig):
    train_dataloader,test_dataloader=dataloader.get_dataloaders(conf)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=AENet(3,1,16).to(device)
    model.load_state_dict(torch.load(conf.infer.infer_model))
    model.eval()
    eval_depth(model,test_dataloader,device,conf)
if __name__ == "__main__":
    main()

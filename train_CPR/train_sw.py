import hydra
from dataloader import sw_dataloader as dataloader
from model.SWnet import SWNET
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


def eval(model,data_loader,conf):
    depth_min,depth_max=conf.smartwatch.depth_min,conf.smartwatch.depth_max
    n_comp_min,n_comp_max=conf.smartwatch.n_comp_min,conf.smartwatch.n_comp_max

    n_error_mean,depth_error_mean=0,0
    for batch in data_loader:
        sw_data, gt_depth, gt_n_comp=batch
        pred_n, pred_depth=model(sw_data)
        pred_n=pred_n*(n_comp_max-n_comp_min)+n_comp_min
        pred_depth=pred_depth*(depth_max-depth_min)+depth_min
        gt_n_comp=gt_n_comp*(n_comp_max-n_comp_min)+n_comp_min
        gt_depth=gt_depth*(depth_max-depth_min)+depth_min

        n_error=torch.abs(pred_n-gt_n_comp)
        depth_error=torch.abs(pred_depth-gt_depth)
        n_error_mean+=n_error.item()
        depth_error_mean+=depth_error.item()
    n_error_mean/=len(data_loader)
    depth_error_mean/=len(data_loader)
    print('------------------------------------')
    print(f'Test : Mean n error: {n_error_mean:.4f} Mean depth error: {depth_error_mean:.4f}')
    print('------------------------------------')

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(conf):
    if conf.smartwatch.get_stats:
        dataloader.get_data_stats(conf)
        return 1
    
    model=SWNET(conf)
    model=model.double()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.smartwatch.lr)
    
    train_dataloader, test_dataloader = dataloader.get_dataloaders(conf)
    for epoch in range(conf.smartwatch.epochs):
        mean_depth_loss,mean_n_loss=0,0 
        for batch in train_dataloader:
            optimizer.zero_grad()
            sw_data, gt_depth, gt_n_comp=batch
            pred_n, pred_depth=model(sw_data)
            n_loss=criterion(pred_n,gt_n_comp)
            depth_loss=criterion(pred_depth,gt_depth.squeeze())
            loss=(n_loss+depth_loss).float()
            loss.backward()
            optimizer.step()
            mean_depth_loss+=depth_loss.item()
            mean_n_loss+=n_loss.item()
        mean_depth_loss/=len(train_dataloader)
        mean_n_loss/=len(train_dataloader)
        print(f'Epoch: {epoch} Mean depth loss: {mean_depth_loss:.4f} Mean n loss: {mean_n_loss:.4f}')
        if (epoch+1)%conf.smartwatch.eval_freq==0:
            eval(model,test_dataloader,conf)
        
if __name__ == "__main__":
    main()
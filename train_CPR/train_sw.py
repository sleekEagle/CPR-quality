import hydra
from dataloader import sw_dataloader as dataloader
from model.SWnet import SWNET
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import utils
import numpy as np
import os

criterion = nn.MSELoss()

def eval(model,data_loader,conf,k=30,height=0.15):
    depth_min,depth_max=conf.smartwatch.depth_min,conf.smartwatch.depth_max
    n_comp_min,n_comp_max=conf.smartwatch.n_comp_min,conf.smartwatch.n_comp_max
    signal_error_mean,depth_error_mean,mean_comp_error=0,0,0
    for batch in data_loader:
        sw_data, gt_depth, gt,gt_n_comp,peaks,valleys=batch
        pred_signal,pred_depth=model(sw_data)
        pred_depth=pred_depth*(depth_max-depth_min)+depth_min
        gt_depth=gt_depth*(depth_max-depth_min)+depth_min

        # plt.plot(gt[0,:].detach().numpy())
        # plt.plot(pred_signal.detach().numpy())
        #detect peaks
        pred=pred_signal.detach().numpy()
        pred=(pred-min(pred))/(max(pred)-min(pred))
        pred=pred-np.mean(pred)
        t=conf.smartwatch.window_len
        num_zero_crossings = len(np.where(np.diff(np.sign(pred)))[0])/t
        dist=int(1/num_zero_crossings*k)
        # plt.plot(pred)
        pred_peaks, pred_valleys,_=utils.find_peaks_and_valleys(pred,distance=dist,height=height,plot=False)
        n_compressions=0.5*(len(pred_peaks)+len(pred_valleys))
        gt_compressions=0.5*(len(peaks[peaks==1])+len(valleys[valleys==1]))
        comp_error=abs(n_compressions-gt_compressions)
        mean_comp_error+=comp_error

        signal_loss=criterion(pred_signal,gt)
        depth_loss=criterion(pred_depth,gt_depth)
        signal_error_mean+=signal_loss.item()
        depth_error_mean+=depth_loss.item()

        
    signal_error_mean/=len(data_loader)
    depth_error_mean/=len(data_loader)
    mean_comp_error/=len(data_loader)

    
    print('------------------------------------')
    print(f'Test : Mean signal error: {signal_error_mean} Mean depth error: {depth_error_mean**0.5:.4f} mm Mean n_comp_error: {mean_comp_error:.4f}')
    print('------------------------------------')

def train(conf):

    model=SWNET(conf)
    model.train()
    model=model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.smartwatch.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    #create directory to save model
    if not os.path.exists(conf.smartwatch.model_save_path):
        os.makedirs(conf.smartwatch.model_save_path,exist_ok=True)

    train_dataloader, test_dataloader = dataloader.get_dataloaders(conf)
    for epoch in range(conf.smartwatch.epochs):
        mean_depth_loss,mean_signal_loss=0,0 
        for batch in train_dataloader:
            optimizer.zero_grad()
            sw_data, gt_depth, gt,gt_n_comp,peaks,valleys=batch
            pred_signal,pred_depth=model(sw_data)
            signal_loss=criterion(pred_signal,gt)
            depth_loss=criterion(pred_depth,gt_depth)
            # pred_valleys=criterion(pred_valleys,valleys)
            # depth_loss=criterion(pred_depth,gt_depth.squeeze())
            loss=signal_loss+depth_loss
            loss.backward()
            optimizer.step()
            
            mean_depth_loss+=depth_loss.item()
            mean_signal_loss+=signal_loss.item()
        mean_depth_loss/=len(train_dataloader)
        mean_signal_loss/=len(train_dataloader)
        print(f'Epoch: {epoch} Mean depth loss: {mean_depth_loss:.4f} Mean signal loss: {mean_signal_loss:.4f}')
        if (epoch+1)%conf.smartwatch.eval_freq==0:
            eval(model,test_dataloader,conf)
            torch.save(model.state_dict(), conf.smartwatch.model_save_path)
        # Step the scheduler
        scheduler.step()

def get_stats(conf):
    if conf.smartwatch.get_stats:
        dataloader.get_data_stats(conf)
   
def eval_checkpt(conf):
    model=SWNET(conf)
    model=model.double()
    model.load_state_dict(torch.load(conf.smartwatch.eval_settings.checkpoint_path))
    model.eval()

    _, test_dataloader = dataloader.get_dataloaders(conf)
    eval(model,test_dataloader,conf,k=conf.smartwatch.eval_setitngs.k,height=conf.smartwatch.eval_setitngs.height)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(conf):
    if conf.smartwatch.mode=='train':
        train(conf)
    elif conf.smartwatch.mode=='eval':
        eval_checkpt(conf)
    elif conf.smartwatch.mode=='stats':
        get_stats(conf)
        
if __name__ == "__main__":
    main()
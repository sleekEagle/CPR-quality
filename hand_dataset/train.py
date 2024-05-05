import hydra
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import dataloader.dataloader as dataloader
from models.Handblur import AENet
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import torch.nn as nn   
import logging
import eval
# Set up logging


torch.manual_seed(2024)
torch.cuda.manual_seed(2024)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    #logging
    logging.basicConfig(filename=f'blur_train_{conf.crop_size}.log', level=logging.INFO)
    # Add the following line at $PLACEHOLDER$
    logging.info('starting....')
    #setup dataset
    train_dataloader,test_dataloader=dataloader.get_dataloaders(conf)

    #setup model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=AENet(3,1,16).to(device)
    model_params=model.parameters()

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model_params, lr=conf.lr)
    scheduler = StepLR(optimizer,step_size=conf.opt_scheduler_step, gamma=0.5)

    #train
    for epoch in range(conf.train_epochs):
        depth_loss,blur_loss=0,0
        for i, batch in enumerate(train_dataloader):
            print(f"training {i} of {len(train_dataloader)}",end='\r')
            img,depth,blur,seg=batch 
            img,depth,blur,seg=img.to(device),depth.to(device),blur.to(device),seg.to(device)
            img=img.swapaxes(2,3).swapaxes(1,2)
            pred_depth, pred_blur,_=model(img)

            optimizer.zero_grad()
            mask=(seg>0) & (depth>0)
            d_loss = criterion(pred_depth.squeeze(dim=1)[mask], depth[mask])
            b_loss=criterion(pred_blur.squeeze(dim=1)[mask], blur[mask])
            loss=d_loss+conf.lmbd*b_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=conf.norm_clip, norm_type=2)
            optimizer.step()

            depth_loss += d_loss.item()
            blur_loss += b_loss.item()

        print(f"Epoch {epoch} depth loss: {depth_loss/len(train_dataloader)} blur loss: {blur_loss/len(train_dataloader)}")
        logging.info(f"Epoch {epoch} depth loss: {depth_loss/len(train_dataloader)} blur loss: {blur_loss/len(train_dataloader)}")
        scheduler.step()
        print(f'last lr: {scheduler.get_last_lr()}')
        if (epoch+1)%conf.eval_freq==0:
            error=eval.eval_depth(model,test_dataloader,device,conf)
            print(f'eval error: {error:.4f}')
            logging.info(f'eval error: {error:.4f}')
            model.train()
            #save model
            torch.save(model.state_dict(), os.path.join(conf.chekpt_path,f'cpr_blur_model_{conf.crop_size}.pth'))

if __name__ == "__main__":
    main()
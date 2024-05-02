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

torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

def eval(model,test_dataloader,device,conf):
    total_error=0
    for i, batch in enumerate(test_dataloader):
        print(f"evaluating {i} of {len(test_dataloader)}",end='\r')
        img,depth,blur,seg=batch 
        img,depth,blur,seg=img.to(device),depth.to(device),blur.to(device),seg.to(device)
        img=img.swapaxes(2,3).swapaxes(1,2)
        pred_depth, pred_blur,_=model(img)

        mask=(seg>0) & (depth>0)
        pred_depth_actual=pred_depth*conf.max_depth
        GT_depth_actual=depth*conf.max_depth
        rmse_error=torch.square(torch.mean((pred_depth_actual.squeeze()[mask]-GT_depth_actual[mask])**2))
        total_error+=rmse_error
    return total_error/len(test_dataloader)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    #setup dataset
    train_dataloader,test_dataloader=dataloader.get_dataloaders(conf)

    #setup model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=AENet(3,1,16).to(device)
    model_params=model.parameters()

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model_params, lr=conf.lr)
    scheduler = StepLR(optimizer,step_size=500, gamma=0.5)

    #train
    for epoch in range(conf.train_epochs):
        model.train()
        depth_loss,blur_loss=0,0
        error=eval(model,test_dataloader,device,conf)
        for i, batch in enumerate(train_dataloader):
            img,depth,blur,seg=batch 
            img,depth,blur,seg=img.to(device),depth.to(device),blur.to(device),seg.to(device)
            img=img.swapaxes(2,3).swapaxes(1,2)
            pred_depth, pred_blur,_=model(img)

            optimizer.zero_grad()
            mask=(seg>0) & (depth>0)
            d_loss = criterion(pred_depth.squeeze()[mask], depth[mask])
            b_loss=criterion(pred_blur.squeeze()[mask], blur[mask])
            loss=d_loss+conf.lmbd*b_loss
            loss.backward()
            optimizer.step()

            depth_loss += d_loss.item()
            blur_loss += b_loss.item()








if __name__ == "__main__":
    main()
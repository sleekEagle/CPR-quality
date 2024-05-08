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
import json
import cv2
import random
import numpy as np

torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

def center_crop(img, output_size):
    width, height,_ = img.shape
    
    # Calculate coordinates for the crop
    center=width//2,height//2
    left=max(0,center[0] - output_size//2)
    right=min(center[0] + output_size//2,width)

    top=max(0,center[1] - output_size//2)
    bottom=min(center[1] + output_size//2,height)

    cropped_image = img[left:right,top:bottom, :]
    return cropped_image, (left, top, right, bottom)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    transform=dataloader.get_transforms(conf,'test')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=AENet(3,1,16).to(device)
    model.load_state_dict(torch.load(conf.infer.infer_model))
    if conf.infer.eval:
        train_dataloader,test_dataloader=dataloader.get_dataloaders(conf)
        eval.eval_depth(model,test_dataloader,device,conf)
    
    data_root=conf.infer.dataset_path
    part_dirs=utils.get_dirs_with_str(data_root,'P',i=0,j=1)
    for p in part_dirs:
        ses_dirs=utils.get_dirs_with_str(p,'s',i=0,j=1)
        for s in ses_dirs:
            print(f'Processing {s}...')
            # img_dir=[s for s in utils.list_subdirectories(s) if (s.startswith('img') or s.startswith('MVI'))][0]
            img_dir=os.path.join(s,'color')
            imgs=utils.list_files(os.path.join(s,img_dir),'jpg')
            bb_path=os.path.join(s,'hand_bbs.json')
            if not os.path.exists(bb_path):
                print(f'{bb_path} does not exist. Continuing...')
                continue
            with open(bb_path, 'r') as f:
                data = json.load(f)

            for img in imgs:
                bbs=data[img.split('.')[0]]
                bbs=[int(item) for item in bbs.split(',')]
                color_img=cv2.imread(os.path.join(os.path.join(s,img_dir,img)))
                center=int(0.5*(bbs[0]+bbs[2])),int(0.5*(bbs[1]+bbs[3]))
                x_crop=utils.get_crop_loc(center[0],0,color_img.shape[1],conf.crop_size)
                y_crop=utils.get_crop_loc(center[1],0,color_img.shape[0],conf.crop_size)
                img_cropped=utils.crop_img_bb(color_img,[x_crop[0],y_crop[0],x_crop[1],y_crop[1]],0)
                img_cropped=transform(image=img_cropped)['image']

                inp=torch.tensor(img_cropped).unsqueeze(0).swapaxes(2,3).swapaxes(1,2).to(device)
                depth_pred,_,_=model(inp)
                real_depth=depth_pred*conf.max_depth
                real_depth=real_depth.squeeze(0).squeeze(0).detach().cpu().numpy()

                depth_img=np.zeros((color_img.shape[0],color_img.shape[1])).astype(np.uint16)
                depth_img[y_crop[0]:y_crop[1],x_crop[0]:x_crop[1]]=(real_depth*1000).astype(np.uint16)
                out_dir=os.path.join(conf.infer.save_path,os.path.basename(p),os.path.basename(s))
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_path=os.path.join(out_dir,img.split('.')[0]+'.png')
                cv2.imwrite(out_path,depth_img)            

                # crop_size=conf.crop_size
                # step=int(conf.crop_size/2)
                # h,w=img_cropped.shape[0],img_cropped.shape[1]
                # h_n,h_rem=h//crop_size,h%crop_size
                # w_n,w_rem=w//crop_size,w%crop_size
                # h_step=crop_size-(crop_size*(h_n+1)-h)//h_n
                # w_step=crop_size-(crop_size*(w_n+1)-w)//w_n
                # h_start=list(range(0,h-crop_size+2,h_step))
                # w_start=list(range(0,w-crop_size+2,w_step))
                # if w_start[-1]+crop_size>w:
                #     w_start[-1]=w_start[-1]-(w_start[-1]+crop_size-w)
                # if h_start[-1]+crop_size>h:
                #     h_start[-1]=h_start[-1]-(h_start[-1]+crop_size-h)

                # n_depths=torch.zeros(h,w)
                # depth_pred_total=torch.zeros(h,w)
                # for h_ in h_start:
                #     for w_ in w_start:
                #         img=img_cropped[h_:h_+crop_size,w_:w_+crop_size,:]
                #         n_depths[h_:h_+crop_size,w_:w_+crop_size]+=1
                #         # img = pad_transform(image=img)['image']
                #         img=torch.tensor(img).unsqueeze(0).swapaxes(2,3).swapaxes(1,2).to(device)
                #         depth_pred,_,_=model(img)
                #         real_depth=depth_pred*conf.max_depth
                #         depth_pred_total[h_:h_+crop_size,w_:w_+crop_size]+=real_depth.squeeze(0).squeeze(0).detach().cpu()
                # depth_pred_total=depth_pred_total/n_depths
                # plt.imshow(depth_pred_total.cpu().numpy())   

if __name__ == "__main__":
    main()







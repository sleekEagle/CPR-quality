import torch
import hydra
from omegaconf import DictConfig
import dataloader.dataloader as dataloader
from models.Handblur import AENet
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import matplotlib.pyplot as plt

torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

def eval_depth_blur(model,test_dataloader,device,conf):
    import numpy as np
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
            # print(torch.mean(depth[depth>0]))
            GT_depth_actual=depth*conf.max_depth
            # print(torch.mean(GT_depth_actual[GT_depth_actual>0]))
            rmse_error=torch.square(torch.mean((pred_depth_actual.squeeze(dim=1)[mask]-GT_depth_actual[mask])**2))
            total_error+=rmse_error.item()
            rmse_errors.append(rmse_error.item())
            gt_dists.append(torch.mean(GT_depth_actual[mask]).item())
    acc=total_error/len(test_dataloader)
    print(f'Eval error: {acc*1000:.2f} mm')
    gt_dists=np.array(gt_dists)
    sort_idx=np.argsort(gt_dists)
    gt_dists=gt_dists[sort_idx]
    rmse_errors=np.array(rmse_errors)[sort_idx]
    print(f'Eval error: {np.array(rmse_errors*1000).mean():.2f} mm')
    print(f'max depth: {gt_dists.max()}')
    arg1=np.where(gt_dists*1000<1000)
    print(f'Eval error <1m: {np.mean(rmse_errors[arg1])*1000}')
    # blur_errors=np.array([gt_dists,rmse_errors])
    # plt.plot(gt_dists,rmse_errors)
    # np.save('gt_dists.npy',blur_errors)
    return acc

def eval_depth_anything(depth_GT_path,depth_anything_path,test_dataloader,conf):
    import cv2
    import numpy as np
    gt_files=utils.list_files(os.path.join(depth_GT_path,'depth'),'png')
    rmse_errors=[]
    gt_dist=[]
    # signed_error=[]
    for gt_file in gt_files:
        depth_gt=cv2.imread(os.path.join(depth_GT_path,'depth',gt_file),cv2.IMREAD_UNCHANGED)
        seg=cv2.imread(os.path.join(depth_GT_path,'seg',gt_file),cv2.IMREAD_UNCHANGED)
        depth_pred=cv2.imread(os.path.join(depth_anything_path,gt_file),cv2.IMREAD_UNCHANGED)
        depth_pred = cv2.resize(depth_pred, (depth_gt.shape[1], depth_gt.shape[0]))
        #get error
        mask=(seg>0) & (depth_gt>0)
        # signed_error.append(np.array(depth_pred[mask>0]-depth_gt[mask>0]).mean())
        rmse_error=np.sqrt(np.mean((depth_gt[mask>0]-(depth_pred[mask>0]))**2))
        rmse_errors.append(rmse_error)
        gt_dist.append(np.mean(depth_gt[mask>0]))
        # print(rmse_error)
    rmse= np.array(rmse_errors).mean()
    gt_dist=np.array(gt_dist)
    sort_idx=np.argsort(gt_dist)
    gt_dist=gt_dist[sort_idx]
    rmse_errors=np.array(rmse_errors)[sort_idx]
    # plt.plot(gt_dist,rmse_errors)
    # depthanything_errors=np.array([gt_dist,rmse_errors])
    # np.save('depthanything_errors.npy',depthanything_errors)
    print(f'Eval error: {np.array(rmse_errors).mean():.2f} mm')
    print(f'max depth: {gt_dist.max()}')
    arg1=np.where(gt_dist<2000)
    print(f'Eval error <1m: {np.mean(rmse_errors[arg1])}')
    return rmse

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig):
    train_dataloader,test_dataloader=dataloader.get_dataloaders(conf)
    if conf.eval_method=='blur':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model=AENet(3,1,16).to(device)
        model.load_state_dict(torch.load(conf.infer.infer_model))
        model.eval()
        eval_depth_blur(model,test_dataloader,device,conf)
    elif conf.eval_method=='depth-anything':
        gt_path=os.path.join(conf.root,'canon')
        eval_depth_anything(gt_path,conf.depthanything_path,test_dataloader,conf)
if __name__ == "__main__":
    main()



#plot 
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter


# deptanything_errors=np.load('depthanything_errors.npy')
# depth_anything_depths=deptanything_errors[0]
# depth_anything_errors=deptanything_errors[1]
# blur_errors=np.load('gt_dists.npy')
# blur_depths=blur_errors[0]  
# blur_errors=blur_errors[1]
# cond1=blur_depths>1
# cond2=blur_depths<1.5
# idx=np.where(cond1 & cond2)
# blur_errors[idx].mean()*1000
# #smooth
# depth_anything_errors_smoothed = np.convolve(depth_anything_errors, np.ones(10)/10, mode='same')
# blur_errors_smoothed = np.convolve(blur_errors, np.ones(10)/10, mode='same')
# blur_errors_smoothed = gaussian_filter(blur_errors, sigma=4)
# depth_anything_errors_smoothed = gaussian_filter(depth_anything_errors, sigma=4)


# plt.plot(depth_anything_depths, depth_anything_errors_smoothed, 'o', label='DepthAnything')
# plt.plot(blur_depths*1000,blur_errors*1000,'o',label='blur')
# plt.show()

# args=np.where(blur_depths<1)
# plt.plot(blur_depths[args],blur_errors[args])






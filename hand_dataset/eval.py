import torch

torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

def eval_depth(model,test_dataloader,device,conf):
    total_error=0
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

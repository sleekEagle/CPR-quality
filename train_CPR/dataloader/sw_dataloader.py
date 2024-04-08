import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import utils
import numpy as np
import cv2
import json

# Define your custom dataset class
class SW_dataset(Dataset):
    def __init__(self, conf,mode):
        self.data_root = conf.data_root
        if mode=='train': self.part=conf.smartwatch.train_part
        elif mode=='test': self.part=conf.smartwatch.test_part
        data_path=os.path.join(self.data_root,'smartwatch_dataset')
        part_data=np.load(os.path.join(data_path,'part_data.npy'))
        valid_args = np.argwhere(np.isin(part_data, self.part))
        # gt_data=np.load(os.path.join(data_path,'gt_data.npy'))[valid_args,:]
        self.gt_depth=np.load(os.path.join(data_path,'depth_data.npy'))[valid_args]
        self.sw_data=np.load(os.path.join(data_path,'sw_data.npy'))[valid_args,:,:,:].squeeze()
        self.gt_n_comp=np.load(os.path.join(data_path,'n_comp_data.npy'))[valid_args].squeeze()
        self.len=len(valid_args)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sw_data=self.sw_data[idx,:,:]
        sw_data = np.reshape(sw_data,(9,300))
        gt_depth=self.gt_depth[idx]
        gt_n_comp=self.gt_n_comp[idx]
        return sw_data, gt_depth, gt_n_comp



def get_dataloaders(conf):
    train_dataset = SW_dataset(conf,'train')
    train_dataloader = DataLoader(train_dataset, batch_size=conf.smartwatch.bs, shuffle=True)

    test_dataset = SW_dataset(conf,'test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_dataloader, test_dataloader

    # for batch in train_dataloader:
    #     print(batch)

def get_data_stats(conf):
    train_dataloader, test_dataloader = get_dataloaders(conf)
    
    def get_stats(dataloader):
        sw_data_list,gt_depth_list,gt_n_comp_list=[],[],[]
        for batch in dataloader:
            sw_data, gt_depth, gt_n_comp=batch
            sw_data_list.append(sw_data)
            gt_depth_list.append(gt_depth)
            gt_n_comp_list.append(gt_n_comp)
            
        sw_data_list=np.concatenate(sw_data_list)
        gt_depth_list=np.concatenate(gt_depth_list)
        gt_n_comp_list=np.concatenate(gt_n_comp_list)
        #get stats
        sw_min=sw_data_list.min(axis=0).min(axis=1)
        sw_max=sw_data_list.max(axis=0).max(axis=1)
        depth_min=np.min(gt_depth_list)
        depth_max=np.max(gt_depth_list) 
        n_comp_min=np.min(gt_n_comp_list)
        n_comp_max=np.max(gt_n_comp_list)
        print(f"sw_min: {','.join(map(str, sw_min))}")
        print(f"sw_max: {','.join(map(str, sw_max))}")
        print(f"depth_min: {depth_min}")
        print(f"depth_max: {depth_max}")
        print(f"n_comp_min: {n_comp_min}")
        print(f"n_comp_max: {n_comp_max}")

    print("Train stats:")
    get_stats(train_dataloader)
    print("Test stats:")
    get_stats(test_dataloader)





# # Iterate over the dataloader
# for batch in dataloader:
#     # Process your batch here
#     print(batch)
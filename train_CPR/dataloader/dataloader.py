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
class CPR_dataset(Dataset):
    def __init__(self, conf,mode):
        self.data_root = conf.data_root
        if mode=='train':
            participants=conf.train_participants
        elif mode=='val':
            participants=conf.test_participants
        
        p_paths=[os.path.join(self.data_root,p) for p in participants]
        session_dirs=[]
        for p in p_paths:
            if not os.path.exists(p):
                raise Exception(f"{p} does not exist")
            session_dirs+=utils.get_dirs_with_str(p,'s')
        self.session_dirs=session_dirs
        self.img_width=conf.img_width

    def __len__(self):
        return len(self.session_dirs)

    def __getitem__(self, index):
        session_dir=self.session_dirs[index]
        print('Accessing data from :',session_dir)
        img_dir=os.path.join(session_dir,'kinect')
        depth_dir=os.path.join(img_dir,'depth')
        mask_dir=os.path.join(img_dir,'hand_mask')
        depth_sensor_file=os.path.join(img_dir,'kinect_depth_interp.txt')
        bb_file=os.path.join(img_dir,'hand_bbs.json')
        # Read the JSON file
        with open(bb_file, 'r') as f:
            bb_data = json.load(f)
        ts_file=os.path.join(img_dir,'kinect_ts.txt')
        GT_depth=np.array(utils.read_allnum_lines(depth_sensor_file))
        ts=np.array(utils.read_allnum_lines(ts_file))
        img_files=utils.get_files_with_str(os.path.join(img_dir,'color'),'.jpg')
        img_keys=[os.path.basename(f).split('.')[0] for f in img_files]
        img_keys.sort()
        last_bb=''
        depth_img_list=[]
        hand_mask_list=[]
        for i,k in enumerate(img_keys):
            depth_file=os.path.join(depth_dir,k+'.png')
            depth_img = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
            mask_file=os.path.join(mask_dir,k+'.png')
            hand_mask=cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
            if k in bb_data.keys():
                bb=bb_data[k]
                if bb!='':
                    last_bb=bb
                else:
                    bb=last_bb
            else:
                bb=last_bb
            if bb=='':
                #get next bb
                idx=i
                while(bb==''):
                    idx+=1
                    bb=bb_data[img_keys[idx]]
                last_bb=bb
            if bb=='':
                print('pp')
            bb=[int(b) for b in bb.split(',')]
            center_coord=(bb[0]+bb[2])//2
            depth_img_list.append(depth_img[:,center_coord-self.img_width:center_coord+self.img_width])
            hand_mask_list.append(hand_mask[:,center_coord-self.img_width:center_coord+self.img_width])
        depth_img_ar=np.array(depth_img_list).astype(np.float32)
        mask_img_ar=np.array(hand_mask_list).astype(np.float32)
        return depth_img_ar,mask_img_ar,GT_depth,ts


def get_dataloaders(conf):
    train_dataset = CPR_dataset(conf,'train')
    train_dataloader = DataLoader(train_dataset, batch_size=conf.bs, shuffle=True)

    test_dataset = CPR_dataset(conf,'train')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_dataloader, test_dataloader

    # for batch in train_dataloader:
    #     print(batch)

# # Create your dataset instance
# data_root=r'D:\CPR_extracted'
# dataset = CPR_dataset(data_root)

# # Create the dataloader
# batch_size = 2
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Iterate over the dataloader
# for batch in dataloader:
#     # Process your batch here
#     print(batch)
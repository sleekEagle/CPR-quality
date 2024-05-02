import torch
from torch.utils.data import Dataset, DataLoader
import hydra
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import utils
import cv2
import json
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt


class HandDepthDataset(Dataset):
    def __init__(self, conf,mode):
        self.conf=conf
        self.data_root = os.path.join(conf.root,conf.camera)
        self.crop=conf.crop_size
        self.mode=mode
        if self.mode=='train':
            self.part=conf.train_part
        elif self.mode=='test':
            self.part=conf.test_part
        #get file names
        self.files=utils.list_files(os.path.join(self.data_root,'color'),'jpg')
        if self.mode in ['train','test']:
            self.files=[f.split('.')[0] for f in self.files if f.split('_')[0] in self.part]

        #read bb
        hand_bb_path=os.path.join(self.data_root,'bbs.json')
        with open(hand_bb_path, 'r') as file:
            dict = json.load(file)
        self.bbs = dict

        if self.mode=='train':
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                A.RandomCrop(width=512, height=512),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(23.4359, 31.9525, 44.8473),std=(31.2212, 40.2346, 52.7526),normalization='image_per_channel')
                ])
        elif self.mode=='test':
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                A.CenterCrop(width=512, height=512),
                A.Normalize(mean=(23.4359, 31.9525, 44.8473),std=(31.2212, 40.2346, 52.7526),normalization='image_per_channel')
                ])
        else:
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                A.CenterCrop(width=512, height=512),
                ])


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Implement your logic to load and preprocess data here
        sample = self.files[index].split('.')[0]
        color_img=cv2.imread(os.path.join(self.data_root,'color',sample+'.jpg'))
        depth_img=cv2.imread(os.path.join(self.data_root,'depth',sample+'.png'),cv2.IMREAD_UNCHANGED)
        seg_img=cv2.imread(os.path.join(self.data_root,'seg',sample+'.png'),cv2.IMREAD_UNCHANGED)
        #get bb
        bb=self.bbs[sample]
        #crop
        color_cropped=utils.crop_img_bb(color_img,bb,0)
        depth_img_cropped=utils.crop_img_bb(depth_img,bb,0)
        seg_img_cropped=utils.crop_img_bb(seg_img,bb,0)

        transformed = self.transform(image=color_cropped,masks=[depth_img_cropped,seg_img_cropped])
        img=transformed['image']
        depth=transformed['masks'][0]
        seg=transformed['masks'][1]

        depth=depth.astype(np.float32)/1000.0
        if self.mode!='stats' and self.conf.normalize_depth:
            depth=depth/2.3
        return img,depth,seg
    

def get_dataloaders(conf):
    train_dataset = HandDepthDataset(conf,'train')
    train_dataloader = DataLoader(train_dataset, batch_size=conf.bs, shuffle=True)
    test_dataset = HandDepthDataset(conf,'train')
    test_dataloader = DataLoader(test_dataset, batch_size=conf.bs, shuffle=True)
    return train_dataloader,test_dataloader

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(conf):
    if conf.mode=='stats':
        dataset = HandDepthDataset(conf,'train')
        dataloader = DataLoader(dataset, batch_size=conf.bs, shuffle=True)
        imgs,depths=torch.empty(0),torch.empty(0)
        for i, batch in enumerate(dataloader):
            print(f'{i}/{len(dataloader)} samples processed',end='\r')
            img,depth,seg=batch 
            imgs = torch.cat((imgs,img), dim=0)
            depths = torch.cat((depths, depth), dim=0)
        d=depths[depths>0]
        print(f'mean depth: {d.mean()}')
        print(f'min depth: {d.min()}')
        print(f'max depth: {d.max()}')
        print(f'std depth: {d.std()}')

        
        print(f'img mean: {imgs.view(-1,3).mean(dim=0)}')
        print(f'img std: {imgs.view(-1,3).std(dim=0)}')
    
    elif conf.mode=='normal':
        train_dataset = HandDepthDataset(conf,'train')
        train_dataloader = DataLoader(train_dataset, batch_size=conf.bs, shuffle=True)
        test_dataset = HandDepthDataset(conf,'train')
        test_dataloader = DataLoader(test_dataset, batch_size=conf.bs, shuffle=True)
if __name__ == "__main__":
    main()



''''
*******stats*******

depth:
mean depth: 0.6976866722106934
min depth: 0.15199999511241913
max depth: 2.2119998931884766
std depth: 0.3109252452850342

image:
mean: tensor([23.4359, 31.9525, 44.8473])
std: tensor([31.2212, 40.2346, 52.7526])
'''
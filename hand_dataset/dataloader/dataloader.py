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

def get_transforms(conf, mode):
    if mode == 'train':
        transforms = A.Compose([
            A.PadIfNeeded(min_height=conf.crop_size, min_width=conf.crop_size, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
            A.RandomCrop(width=conf.crop_size, height=conf.crop_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(33.4706, 45.6208, 64.9402), std=(31.5944, 39.3640, 49.9791), normalization='image_per_channel'),
            A.Resize(256,256)
        ])
    elif mode == 'test':
        transforms = A.Compose([
            A.PadIfNeeded(min_height=conf.crop_size, min_width=conf.crop_size, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
            A.CenterCrop(width=conf.crop_size, height=conf.crop_size),
            A.Normalize(mean=(33.4706, 45.6208, 64.9402), std=(31.5944, 39.3640, 49.9791), normalization='image_per_channel'),
            A.Resize(256,256)
        ])
    elif mode == 'stats':
        transforms = A.Compose([
            A.PadIfNeeded(min_height=conf.crop_size, min_width=conf.crop_size, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
            A.RandomCrop(width=conf.crop_size, height=conf.crop_size),
            A.Normalize(mean=(33.4706, 45.6208, 64.9402), std=(31.5944, 39.3640, 49.9791), normalization='image_per_channel')
        ])

    return transforms


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
        if self.conf.data_out!='stats':
            self.transform = get_transforms(conf, self.mode)
        else :
            self.transform = get_transforms(conf, 'stats')


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
        blur=-1
        if self.conf.get_blur:
            blur=np.abs(1/(depth+1e-5)-1)
            blur[blur>1000]=0
            if self.conf.data_out !='stats' and self.conf.normalize_blur:
                blur=blur/self.conf.max_blur
        if self.conf.data_out !='stats' and self.conf.normalize_depth:
            depth=depth/self.conf.max_depth
        return img,depth,blur,seg
    

def get_dataloaders(conf):
    train_dataset = HandDepthDataset(conf,'train')
    train_dataloader = DataLoader(train_dataset, batch_size=conf.bs, shuffle=True)
    test_dataset = HandDepthDataset(conf,'test')
    test_dataloader = DataLoader(test_dataset, batch_size=conf.bs, shuffle=True)
    return train_dataloader,test_dataloader

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(conf):
    if conf.data_out=='stats':
        dataset = HandDepthDataset(conf,'other')
        dataloader = DataLoader(dataset, batch_size=conf.bs, shuffle=True)
        imgs,depths,blurs=torch.empty(0),torch.empty(0),torch.empty(0)
        for i, batch in enumerate(dataloader):
            print(f'{i}/{len(dataloader)} samples processed',end='\r')
            img,depth,blur,seg=batch 
            imgs = torch.cat((imgs,img), dim=0)
            depths = torch.cat((depths, depth), dim=0)
            blurs = torch.cat((blurs, blur), dim=0)
        d=depths[depths>0]
        b=blurs[blurs>0]
        print(f'mean depth: {d.mean()}')
        print(f'min depth: {d.min()}')
        print(f'max depth: {d.max()}')
        print(f'std depth: {d.std()}')

        print(f'mean blur: {b.mean()}')
        print(f'min blur: {b.min()}')
        print(f'max blur: {b.max()}')
        print(f'std blur: {b.std()}')


        d_np=d.cpu().numpy()
        plt.hist(d_np,color='#10439F',bins=50)
        plt.xlabel('Depth (m)')
        plt.ylabel('Frequency')
        plt.savefig(r'C:\Users\lahir\Downloads\depth_dataset_stats.png', dpi=600)
        plt.show()


        
        print(f'img mean: {imgs.view(-1,3).mean(dim=0)}')
        print(f'img std: {imgs.view(-1,3).std(dim=0)}')
    
    else:
        train_dataset = HandDepthDataset(conf,'train')
        train_dataloader = DataLoader(train_dataset, batch_size=conf.bs, shuffle=True)
        test_dataset = HandDepthDataset(conf,'test')
        test_dataloader = DataLoader(test_dataset, batch_size=conf.bs, shuffle=True)
if __name__ == "__main__":
    main()



''''
*******stats*******
-----------*Canon dataset*-------
depth:
mean depth: 0.6976866722106934
min depth: 0.15199999511241913
max depth: 2.2119998931884766
std depth: 0.3109252452850342

blur:
mean blur: 0.7680147886276245
min blur: 1.0013580322265625e-05
max blur: 5.710958957672119
std blur: 0.6914953589439392


image:
mean: tensor([23.4359, 31.9525, 44.8473])
std: tensor([31.2212, 40.2346, 52.7526])
'''

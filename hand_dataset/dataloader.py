import torch
from torch.utils.data import Dataset, DataLoader
import hydra
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import cv2

class HandDepthDataset(Dataset):
    def __init__(self, conf,mode):
        self.data_root = os.path.join(conf.root,conf.camera)
        self.crop=conf.crop_size
        if mode=='train':
            self.part=conf.train_part
        elif mode=='test':
            self.part=conf.test_part
        #get file names
        files=utils.list_files(os.path.join(self.data_root,'color'),'jpg')
        self.files=[f.split('.')[0] for f in files if f.split('_')[0] in self.part]

        #read bb
        hand_bb_path=os.path.join(self.data_root,'bbs.txt')
        with open(hand_bb_path, 'r') as file:
            data = file.readlines()
        f_bb_list=[(line.strip().split(',')[0],line.strip().split(',')[1:])  for line in data]
        self.bbs={}
        for f,bb in f_bb_list:
            bb= [int(b) for b in bb]
            self.bbs[f]=bb
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        import matplotlib.pyplot as plt
        # Implement your logic to load and preprocess data here
        sample = self.files[index]
        color_img=cv2.imread(os.path.join(self.data_root,'color',sample+'.jpg'))
        depth_img=cv2.imread(os.path.join(self.data_root,'depth',sample+'.png'),cv2.IMREAD_UNCHANGED)
        seg_img=cv2.imread(os.path.join(self.data_root,'seg',sample+'.png'),cv2.IMREAD_UNCHANGED)
        #get bb
        bb=self.bbs[sample]
        #crop
        

        
        


        color_cropped=color_img[bb[1]:bb[3],bb[0]:bb[2],:]
        depth_cropped=depth_img[bb[1]:bb[3],bb[0]:bb[2]]
        seg_cropped=seg_img[bb[1]:bb[3],bb[0]:bb[2]]

        return color_cropped.shape



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    dataset = HandDepthDataset(conf,'train')
    dataloader = DataLoader(dataset, batch_size=conf.bs, shuffle=True)
    x,y=torch.empty(0),torch.empty(0)
    for i, batch in enumerate(dataloader):
        s=batch 
        x = torch.cat((x, s[0]), dim=0)
        y = torch.cat((y, s[1]), dim=0)
    pass
if __name__ == "__main__":
    main()

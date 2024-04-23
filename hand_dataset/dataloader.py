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
        if mode=='train':
            self.part=conf.train_part
        elif mode=='test':
            self.part=conf.test_part
        #get file names
        files=utils.list_files(os.path.join(self.data_root,'color'),'jpg')
        self.files=[f.split('.')[0] for f in files if f.split('_')[0] in self.part]

        #read bb
        # hand_bb_path=os.path.join(self.data_root,'bbs.txt')
        # with open(hand_bb_path, 'r') as file:
        #     data = file.readlines()
        # [line.strip().split(',') for line in data][0]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Implement your logic to load and preprocess data here
        sample = self.files[index]
        color_img=cv2.imread(os.path.join(self.data_root,'color',sample+'.jpg'))
        depth_img=cv2.imread(os.path.join(self.data_root,'depth',sample+'.png'),cv2.IMREAD_UNCHANGED)
        seg_img=cv2.imread(os.path.join(self.data_root,'seg',sample+'.png'),cv2.IMREAD_UNCHANGED)
        # Return the preprocessed sample
        return sample



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    print('here')
    dataset = HandDepthDataset(conf,'train')
    dataloader = DataLoader(dataset, batch_size=conf.bs, shuffle=True)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched)
        break

if __name__ == "__main__":
    main()

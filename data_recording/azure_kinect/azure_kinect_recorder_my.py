# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/sensors/azure_kinect_recorder.py

import argparse
import datetime
import open3d as o3d
from datetime import datetime
from os.path import join
import numpy as np
import os
from PIL import Image
import time

'''
We modified this so the record filename includes the timestamp when the recording is started. 
(when the space bar is pressed)
'''

class RecorderWithCallback:

    def __init__(self, config, device, record_filename, txt_filename, align_depth_to_color,n):
        # Global flags
        self.flag_exit = False
        self.flag_record = False
        self.filename = record_filename
        self.ts_filename=txt_filename
        self.n=n

        self.align_depth_to_color = align_depth_to_color
        self.recorder = o3d.io.AzureKinectRecorder(config, device)
        if not self.recorder.init_sensor():
            raise RuntimeError('Failed to connect to sensor')
        
    def get_ts(self):
        # t=datetime.now()
        # now_str = t.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        # now_str=now_str.split(' ')[-1]
        t=str(time.time_ns())
        # return now_str.replace(':','_') 
        return t


    def escape_callback(self, vis):
        self.flag_exit = True
        if self.recorder.is_record_created():
            print('Recording finished.')
        else:
            print('Nothing has been recorded.')
        return False

    def space_callback(self, vis):
        if self.flag_record:
            print('Recording paused. '
                  'Press [Space] to continue. '
                  'Press [ESC] to save and exit.')
            self.flag_record = False

        elif not self.recorder.is_record_created():   
                     
            if self.recorder.open_record(self.filename):
                print('Recording started. '
                      'Press [SPACE] to pause. '
                      'Press [ESC] to save and exit.')
                self.flag_record = True

        else:
            print('Recording resumed, video may be discontinuous. '
                  'Press [SPACE] to pause. '
                  'Press [ESC] to save and exit.')
            self.flag_record = True

        return False

    def run(self):
        glfw_key_escape = 256
        glfw_key_space = 32
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.register_key_callback(glfw_key_space, self.space_callback)

        vis.create_window('recorder', 1920, 540)
        print("Recorder initialized. Press [SPACE] to start. "
              "Press [ESC] to save and exit.")

        vis_geometry_added = False
        num_imgs=0
        
        while not self.flag_exit:
            rgbd = self.recorder.record_frame(self.flag_record,
                                              self.align_depth_to_color)
            if rgbd is None:
                continue

            if self.flag_record:
                #write ts to the ts file
                ts=self.get_ts()
                with open(self.ts_filename, 'a') as f:
                    f.write(ts+'\n')
        
                if not self.n == -1:
                    num_imgs+=1
                    if num_imgs==self.n:
                        break

            if not vis_geometry_added:
                vis.add_geometry(rgbd)
                vis_geometry_added = True

            vis.update_geometry(rgbd)
            vis.poll_events()
            vis.update_renderer()

        self.recorder.close_record()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Azure kinect mkv recorder.')
    parser.add_argument('--config', type=str, help='input json kinect config')
    parser.add_argument('--output', type=str,
                        default='C:\\Users\\lahir\\data\\kinect_hand_data\\test\\',
                        help='output directory')
    parser.add_argument('--list',
                        action='store_true',
                        help='list available azure kinect sensors')
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='input kinect device id')
    parser.add_argument('--n',
                        type=int,
                        default=-1,
                        help='number of images to take. -1 to ercord until ESC is pressed')
    parser.add_argument('-a',
                        '--align_depth_to_color',
                        action='store_true',
                        help='enable align depth image to color')
    args = parser.parse_args()

    SIZE=(1280,720)

    if args.list:
        o3d.io.AzureKinectSensor.list_devices()
        exit()

    if args.config is not None:
        print(f'config path: {args.config}')
        config = o3d.io.read_azure_kinect_sensor_config(args.config)
    else:
        config = o3d.io.AzureKinectSensorConfig()

    assert args.output is not None , "output must be given"
    ts='{date:%Y-%m-%d-%H-%M-%S}.mkv'.format(
        date=datetime.now())
    record_filename = join(args.output,ts)
    print('Prepare writing to {}'.format(record_filename))

    #generate text file to write the timestamps
    ts_txt=ts.replace('.mkv','.txt')
    txt_filename = join(args.output,ts_txt)
    print('Creating ts file {}'.format(txt_filename))

    device = args.device
    if device < 0 or device > 255:
        print('Unsupported device id, fall back to 0')
        device = 0

    r = RecorderWithCallback(config, device, record_filename, txt_filename,
                             args.align_depth_to_color,args.n)
    r.run()

    #read the mkv file and extract images
    # reader = o3d.io.AzureKinectMKVReader()
    # reader.open(record_filename)
    # print('opened')

    # color_dir=os.path.join(args.output,'color')
    # depth_dir=os.path.join(args.output,'depth')
    # print(color_dir)
    # print(depth_dir)
    # try:
    #     os.makedirs(color_dir, exist_ok=False)
    #     os.makedirs(depth_dir, exist_ok=False)
    # except Exception as e:
    #     print("Error:", e)
    
    # #read existing color images
    # img_n=len([file for file in os.listdir(color_dir) if file.split('.')[-1].lower()=='jpg'])
    # print(f'n images = {img_n}')

    # while not reader.is_eof():
    #     rgbd = reader.next_frame()
    #     if rgbd is None:
    #         continue
    #     if args.output is not None:
    #         print('here')
    #         img_n+=1
    #         np_img=np.array(rgbd.color)
    #         print(f'image size {np_img.shape}')
    #         col_image=Image.fromarray(np_img,'RGB')
    #         #pad image
    #         result = Image.new(col_image.mode, SIZE, (0, 0, 0)) 
    #         result.paste(col_image, (0,0)) 
    #         result.save(os.path.join(color_dir,str(img_n)+'.jpg'))
    #         depthimg=Image.fromarray(np.asarray(rgbd.depth))
    #         depthimg.save(os.path.join(depth_dir,str(img_n)+'.png'))
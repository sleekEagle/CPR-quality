import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.linear_model import RANSACRegressor
import matplotlib.animation as animation
import utils


path='D:\\CPR_data_raw\\P0\\extracted\\s_0\\'
img_dir=os.path.join(path,'color')
depth_dir=os.path.join(path,'depth')
ts_file=os.path.join(path,'kinect_ts.txt')
with open(ts_file, 'r') as file:
    # Read all lines into a list
    ts_lines = file.readlines()
ts_list=np.array([float(line.strip()) for line in ts_lines])

k=np.array([[615.873673811006,0,640.803032851225],[0,615.918359977960,365.547839233105],[0,0,1]])

img_key='01508'

def get_XYZ(x,y,depth,k):
    X=x*depth/k[0,0]
    Y=y*depth/k[1,1]
    Z=depth
    return X,Y,Z

#read keypoint data
with open(os.path.join(path,'wrist_keypts','hand_keypts_mediapipe.json'), 'r') as json_file:
    kypt_dict = json.load(json_file)

img_list=os.listdir(img_dir)
img_list.sort()
XYZ_list=[]
for img in img_list:
    if img.endswith('.jpg'):
        img_key=img.split('.')[0]
        depth_file=os.path.join(depth_dir,img_key+'.png')
        depth_img = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        kypts=kypt_dict[img_key]
        if len(kypts)>0:
            x,y=kypts['x'][0],kypts['y'][0]
            wrist_depth=np.median(depth_img[y-5:y+5,x-5:x+5])
            depth=wrist_depth
            X,Y,Z=get_XYZ(x,y,depth,k)
            XYZ_list.append([X,Y,Z])
        else:
            XYZ_list.append([-1,-1,-1])

t=np.arange(0,len(XYZ_list))

XYZ_list_cleaned=[]
t_selected_list=[]
for i,XYZ in enumerate(XYZ_list):
    if XYZ[0]>0:
        XYZ_list_cleaned.append(XYZ)
        t_selected_list.append(ts_list[i])
XYZ_list_cleaned=np.array(XYZ_list_cleaned)
t_selected_list=np.array(t_selected_list)

#interpolate points
first_t=ts_list[0]
input_ts=t_selected_list-first_t
output_ts=ts_list-first_t
last_valid_idx=np.max(np.argwhere(output_ts<input_ts[-1]))
output_ts=output_ts[:last_valid_idx]

out_interp=utils.interpolate_between_ts(XYZ_list_cleaned[:,1],input_ts,output_ts,fit_window=50,deg=4)

input_ts[0],input_ts[-1]
output_ts[0],output_ts[-1]

plt.plot(input_ts,XYZ_list_cleaned[:,1])
plt.plot(output_ts,out_interp)


# plt.scatter(t_selected_list-first_t,XYZ_list_cleaned[:,2],s=5)
# plt.show()

# utils.animate_pt_seq(XYZ_list_cleaned, interval=0.01)


# t_list=np.array([t_list]).T

# plt.scatter(XYZ_list_cleaned[:,0],XYZ_list_cleaned[:,1],XYZ_list_cleaned[:,2])
# plt.show()

# ransac = RANSACRegressor()
# ransac.fit(XYZ_list_cleaned[:,0:2], XYZ_list_cleaned[:,2])
# pred=ransac.predict(XYZ_list_cleaned[:,0:2])

# inlier_mask = ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)


# plt.plot(XYZ_list_cleaned[:,2])
# plt.plot(pred)
# plt.show()

# from matplotlib.animation import FuncAnimation

# def update_graph(num):
#     data=XYZ_list_cleaned[num,:]
#     graph._offsets3d = (data[0], data[1], data[2])
#     title.set_text('3D Test, time={}'.format(num))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# title = ax.set_title('3D Test')

# data=XYZ_list_cleaned[0,:]
# graph = ax.scatter(data[0], data[1], data[2])
# ani = FuncAnimation(fig, update_graph, 19, 
#                                interval=40, blit=False)
# plt.show()



# from sklearn import datasets, linear_model
# X, y, coef = datasets.make_regression(
#     n_samples=100,
#     n_features=1,
#     n_informative=1,
#     noise=10,
#     coef=True,
#     random_state=0,
# )





# XYZ_list
# x_vals=np.array([XYZ[0] for XYZ in XYZ_list])
# y_vals=np.array([XYZ[1] for XYZ in XYZ_list])
# z_vals=np.array([XYZ[2] for XYZ in XYZ_list])

# x_vals=x_vals[x_vals>0]
# y_vals=y_vals[y_vals>0] 
# z_vals=z_vals[z_vals>0] 


# plt.plot(y_vals[y_vals>0])
# plt.show()

# first_point=XYZ_list[0]
# for i in range(1,len(XYZ_list)):
#     if XYZ_list[i]==[-1]:
#         XYZ_list[i]=first_point
#     else:
#         first_point=XYZ_list[i]

# ransac = RANSACRegressor()
# ransac.fit(np.array(XYZ_list), np.array(XYZ_list))


# depth_list=np.array(depth_list)
# depth_list=depth_list[depth_list>0]
# plt.plot(depth_list)
# plt.show()

# depth_files=[os.path.join(depth_dir,item) for item in os.listdir(img_dir) if item.endswith('.jpg')]

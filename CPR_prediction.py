import json
import utils
import os
import numpy as np
import matplotlib.pyplot as plt
import hydra
import cv2
# from sklearn.linear_model import RANSACRegressor
# import pyransac3d as pyrsc
# from scipy.interpolate import interp1d
# from scipy.interpolate import CubicSpline


def project_point_to_plane(point, plane):
    # Plane equation: ax + by + cz + d = 0
    a, b, c, d = plane
    x, y, z = point

    # Normal vector
    N = np.array([a, b, c])

    # Project point onto plane
    projected_point = point - ((a*x + b*y + c*z + d) / np.sum(N**2)) * N

    return projected_point


def dist_from_plane(A,B,C,D,XYZ_array):
    return (A*XYZ_array[:,0]+B*XYZ_array[:,1]+C*XYZ_array[:,2]+D)/np.sqrt(A**2+B**2+C**2)

def read_XYZ(path,XYZ_file):
    kinect_ts_path=os.path.join(path,'kinect_ts_interp.txt')
    XYZ_dict_path=os.path.join(path,'wrist_keypts',XYZ_file)
    kinect_depth_path=os.path.join(path,'kinect_depth_interp.txt')

    #get image file keys    
    img_names=utils.get_files_with_str(os.path.join(path,'color'),'.jpg')
    valid_keys=[os.path.basename(img).split('.')[0] for img in img_names]

    if not os.path.exists(XYZ_dict_path):
        print(f'{XYZ_dict_path} does not exist')
        return -1
    with open(XYZ_dict_path, 'r') as file:
        XYZ_dict = json.load(file)

    kinect_ts_list=utils.read_allnum_lines(kinect_ts_path)    
    sorted_keys=[key for key in XYZ_dict.keys() if key in valid_keys]
    sorted_keys.sort()

    kinect_inter_depth_list=utils.read_allnum_lines(kinect_depth_path)

    XYZ_list=[]
    XYZ_idx_list=[]
    for i,key in enumerate(sorted_keys):
        vals=XYZ_dict[key]
        if len(vals)==0 or vals['0']=='': continue
        if vals=='' : continue
        wrist=[float(v) for v in vals['0']]
        XYZ_list.append(wrist) 
        XYZ_idx_list.append(i) 
    
    XYZ_array=np.array(XYZ_list)
    

    # Detect outliers of depths based on median for depth
    depths=XYZ_array[:,2]
    median_depth = np.median(depths)
    median_absolute_deviation = np.median(np.abs(depths - median_depth))
    threshold = 1.5 * median_absolute_deviation
    outliers = np.abs(depths - median_depth) > threshold

    filtered_depths = depths[~outliers]
    outlier_args=np.argwhere(outliers)
    valid_args=np.argwhere(~outliers)
    all_args=np.arange(len(depths))
    out_ts,valid_out_ts,depth_int=utils.interpolate_between_ts_cube(depths[valid_args][:,0],valid_args[:,0],all_args,plot=False)

    #remove invalid values from data
    XYZ_array=XYZ_array[valid_out_ts,:]
    XYZ_idx_list=np.array(XYZ_idx_list)
    XYZ_idx_list=XYZ_idx_list[valid_out_ts]
    XYZ_array[:,2]=depth_int    

    output={}
    output['XYZ_array']=XYZ_array
    output['kinect_inter_depth_list']=kinect_inter_depth_list
    output['kinect_ts_list']=kinect_ts_list
    output['XYZ_idx_list']=XYZ_idx_list

    return output

def getXYZpos(XYZ_array,XYZ_idx_list,kinect_inter_depth_list,kinect_ts_list,plot=False):
    kinect_ts_list=np.array(kinect_ts_list)
    valid_t=np.array(kinect_ts_list)[XYZ_idx_list]
    ts_x,valid_out_ts,x_pred=utils.interpolate_between_ts_cube(XYZ_array[:,0],valid_t,kinect_ts_list,plot=False)
    ts_y,valid_out_ts,y_pred=utils.interpolate_between_ts_cube(XYZ_array[:,1],valid_t,kinect_ts_list,plot=False)
    ts_z,valid_out_ts,z_pred=utils.interpolate_between_ts_cube(XYZ_array[:,2],valid_t,kinect_ts_list,plot=False)
    assert np.all(ts_x==ts_y) and np.all(ts_x==ts_z) and np.all(ts_y==ts_z), "A problem in interpolation"
    kinect_inter_depth_list=np.array(kinect_inter_depth_list)[valid_out_ts]

    # plt.plot(valid_t,XYZ_array[:,0])
    # plt.plot(ts_x,x_pred)

    XYZ_array_interp=np.array([x_pred,y_pred,z_pred]).T
    #detect peaks and valleys
    mins,maxs=np.min(XYZ_array_interp,axis=0),np.max(XYZ_array_interp,axis=0)
    XYZ_array_norm=(XYZ_array_interp-mins)/(maxs-mins)
    XYS_sqsum=np.sum(XYZ_array_norm**2,axis=1)
    #normalize again
    mins,maxs=np.min(XYS_sqsum),np.max(XYS_sqsum)
    XYS_sqsum_norm=(XYS_sqsum-mins)/(maxs-mins)

    #curve fitting to smoothe the normed data
    t=np.arange(0,len(XYS_sqsum_norm))
    t=t/max(t)
    pred=utils.interpolate_between_ts(XYS_sqsum_norm,t,t,fit_window=50,deg=4)

    peaks,valleys=utils.find_peaks_and_valleys(pred)
    len_peaks=len(peaks)
    len_valleys=len(valleys)
    l=min(len_peaks,len_valleys)
    peaks=peaks[:l]
    valleys=valleys[:l]

    if plot:
        plt.plot(pred)
        plt.plot(peaks, pred[peaks], "x")
        plt.plot(valleys, pred[valleys], "o")
        plt.title('peaks and valleys')
        plt.show()

    # fit_window=50
    # deg=4
    # x_pred=utils.interpolate_between_ts(XYZ_array_interp[:,0],t,t,fit_window=fit_window,deg=deg)
    # y_pred=utils.interpolate_between_ts(XYZ_array_interp[:,1],t,t,fit_window=fit_window,deg=deg)
    # z_pred=utils.interpolate_between_ts(XYZ_array_interp[:,2],t,t,fit_window=fit_window,deg=deg)
    # XYZ_val_pred=np.array([x_pred,y_pred,z_pred])

    l=min(peaks.shape,valleys.shape)[0]
    high_vals=XYZ_array_interp[peaks[:l],:]
    low_vals=XYZ_array_interp[valleys[:l],:]
    depths=np.sum((high_vals-low_vals)**2,axis=1)**0.5
    # plt.plot(depths)

    #get GT depths
    GT_high_vals=kinect_inter_depth_list[peaks[:l]]
    GT_low_vals=kinect_inter_depth_list[valleys[:l]]
    GT_depths=(GT_high_vals-GT_low_vals)

    plt.plot(depths)
    plt.plot(GT_depths)

    output={}
    output['GT_depths']=GT_depths
    output['depths']=depths
    output['peaks']=peaks
    output['valleys']=valleys
    output['interpolated_norm']=pred
    output['kinect_inter_depth_list']=kinect_inter_depth_list
    output['high_vals']=high_vals
    output['low_vals']=low_vals
    
    return output



'''
get xy and timestamps for a given session
'''
def get_xy_times(path,method):
    #read ts file
    ts_vals=utils.read_allnum_lines(os.path.join(path,'kinect_ts_interp.txt'))
    xy_kypt_path=os.path.join(path,'wrist_keypts',f'hand_keypts_{method}.json')
    img_names=[os.path.basename(img).split('.')[0] for img in utils.get_files_with_str(os.path.join(path,'color'),'.jpg')]
    img_names.sort()
    valid_img_names=[]
    kypts=[]
    valid_ts=[]
    with open(xy_kypt_path, 'r') as file:
        xy_kypts = json.load(file)
        for k in xy_kypts.keys():
            if not (k in xy_kypts.keys()):
                continue
            if (len(xy_kypts[k]['x'])==0) or (len(xy_kypts[k]['y'])==0):
                continue
            valid_img_names.append(k)
            kypts.append((xy_kypts[k]['x'][0],xy_kypts[k]['y'][0]))
            valid_ts.append(ts_vals[img_names.index(k)])
    output={}
    output['valid_img_names']=valid_img_names
    output['kypts']=kypts
    output['valid_ts']=valid_ts
    return output


path=r'D:\CPR_extracted\P0\s_0\kinect'

def detect_CPR_rate_depth(path,config):
    #detect peaks and valleys of the wrist movement
    xy_out=get_xy_times(path,config.evaluate.method)
    signal=np.array([k[1] for k in xy_out['kypts']])
    peaks,valleys=utils.detect_peaks_and_valleys_depth_sensor(signal,xy_out['valid_ts'],show=False)

    #detect CPR rate
    p=(len(peaks)+len(valleys))/2
    t=(xy_out['valid_ts'][-1]-xy_out['valid_ts'][0])/60
    CPR_rate=p/t
    print(f"CPR rate is {CPR_rate:.2f} compressions per minute")

    peak_times=np.array(xy_out['valid_ts'])[peaks]
    valley_times=np.array(xy_out['valid_ts'])[valleys]
    xy_vals_peaks=np.array(xy_out['kypts'])[peaks]
    xy_vals_valleys=np.array(xy_out['kypts'])[valleys]
    peak_img_names=[xy_out['valid_img_names'][i] for i in peaks]
    valley_img_names=[xy_out['valid_img_names'][i] for i in valleys]

    peak_depths=np.array([utils.get_depth_val_from_xy(os.path.join(path,'depth',peak_img_names[i]+'.png'),xy_vals_peaks[i][0],xy_vals_peaks[i][0]) for i in range(len(peak_img_names))])
    valley_depths=np.array([utils.get_depth_val_from_xy(os.path.join(path,'depth',valley_img_names[i]+'.png'),xy_vals_valleys[i][0],xy_vals_valleys[i][0]) for i in range(len(valley_img_names))])

    #get XYZ points
    XYZ_peaks=[utils.get_XYZ_kinect(xy_vals_peaks[i][0],xy_vals_peaks[i][1],peak_depths[i]) for i in range(len(peak_img_names))]
    XYZ_valleys=[utils.get_XYZ_kinect(xy_vals_valleys[i][0],xy_vals_valleys[i][1],valley_depths[i]) for i in range(len(valley_img_names))]
    XYZ_peaks=np.array(XYZ_peaks)
    XYZ_valleys=np.array(XYZ_valleys)
    l=min(len(XYZ_peaks),len(XYZ_valleys))
    XYZ_peaks=XYZ_peaks[:l]
    XYZ_valleys=XYZ_valleys[:l]
    CPR_depth= np.sum((XYZ_peaks-XYZ_valleys)**2,axis=1)**0.5
    #remove outliers
    median=np.median(CPR_depth)
    mad=np.median(np.abs(CPR_depth-median))
    outliers = np.abs(CPR_depth - median) / mad > 3
    CPR_depth=CPR_depth[~outliers]
    mean_CPR_depth=np.mean(CPR_depth)
    print(f"Mean CPR depth is {mean_CPR_depth:.2f} mm")

    return mean_CPR_depth,CPR_rate





@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    mean_CPR_depth,CPR_rate=detect_CPR_rate_depth(path,config)
    # method=config.evaluate.method
    # plot_3d=True
    # plot_depth=True
    # root_dir=r'D:\CPR_extracted'
    # method='mediapipe'   #'mediapipe' or 'mmpose_finetuned_RHD2D'
    # XYZ_file=f'hand_keypts_{method}_XYZ.json'
    # output_file=f'performance_{method}.txt'
    # output_file=os.path.join(root_dir,output_file)

    # with open(output_file, 'a') as file:
    #     subj_dirs=utils.get_dirs_with_str(root_dir, 'P')
    #     for subj_dir in subj_dirs:
    #         session_dirs=utils.get_dirs_with_str(subj_dir,'s')
    #         for session_dir in session_dirs:
    #             print(session_dir)
    #             path=os.path.join(session_dir,'kinect')
    #             if not os.path.exists(path):
    #                 file.write(session_dir+','+str(-1))
    #                 file.write("\n")
    #                 print('written')
    #                 continue
    #             init_output=read_XYZ(path,XYZ_file)
    #             if (init_output==-1):
    #                 file.write(session_dir+','+str(-1))
    #                 file.write("\n")
    #                 print('written')
    #                 continue
    #             XYZ_array = init_output['XYZ_array']
    #             kinect_inter_depth_list=init_output['kinect_inter_depth_list']
    #             kinect_ts_list=init_output['kinect_ts_list']
    #             XYZ_idx_list=init_output['XYZ_idx_list']
    #             if plot_3d:
    #                 fig = plt.figure()
    #                 ax = fig.add_subplot(111, projection='3d')
    #                 ax.set_xlim(min(XYZ_array[:,0]), max(XYZ_array[:,0]))
    #                 ax.set_ylim(min(XYZ_array[:,1]), max(XYZ_array[:,1]))
    #                 ax.scatter(XYZ_array[:, 0], XYZ_array[:, 1], XYZ_array[:, 2])
    #                 ax.scatter(XYZ_array[0, 0], XYZ_array[0, 1], XYZ_array[0, 2], s=100, c='red')
    #                 # ax.scatter(XYZ_valid[:, 0], XYZ_valid[:, 1], XYZ_valid[:, 2], s=100, c='yellow')
    #                 # ax.plot_surface(x, y, z, alpha=0.5, rstride=100, cstride=100, color='green')
    #                 ax.set_xlabel('X Axis')
    #                 ax.set_ylabel('Y Axis')
    #                 ax.set_zlabel('Z Axis')
    #                 plt.title('movement of the wrist in 3D space and plane detection')
    #                 plt.show()
    #             output=getXYZpos(XYZ_array,XYZ_idx_list,kinect_inter_depth_list,kinect_ts_list,plot=False)
    #             high_vals=output['high_vals']
    #             low_vals=output['low_vals']
    #             GT_depth=output['GT_depths']
    #             depth_est=output['depths']
    #             if plot_depth:
    #                 plt.plot(GT_depth)
    #                 plt.plot(depth_est)
    #                 plt.show()
    #             error=np.mean(np.abs(GT_depth-depth_est))
    #             print(f"The error is {error:.2f} mm")
    #             file.write(session_dir+','+f"{error:.2f}")
    #             file.write("\n")
    #             print('written')

if __name__ == "__main__":
    main()

#detect planes from data
# plane1 = pyrsc.Plane()
# best_eq, best_inliers = plane1.fit(XYZ_array, 0.01)
# A, B, C, D=best_eq
# x=np.linspace(min(XYZ_array[:,0]),max(XYZ_array[:,0]),100)
# y=np.linspace(min(XYZ_array[:,1]),max(XYZ_array[:,1]),100)
# x, y = np.meshgrid(x, y)
# z = (-D - A * x - B * y) / C



# threshold=5
# dists=np.abs(dist_from_plane(A, B, C, D, XYZ_array))
# dists_valid=dists<threshold
# XYZ_valid=XYZ_array[dists_valid]




#project points onto the plane
# proj_points_high=np.array([project_point_to_plane(val, best_eq) for val in high_vals])
# proj_points_low=np.array([project_point_to_plane(val, best_eq) for val in low_vals])
# depths=np.sum((high_vals-proj_points_low)**2,axis=1)**0.5

# plt.plot(output['GT_depths'])
# plt.plot(depths)
# plt.show()




# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim(min(XYZ_valid[:,0]), max(XYZ_valid[:,0]))
# ax.set_ylim(min(XYZ_valid[:,1]), max(XYZ_valid[:,1]))
# ax.scatter(XYZ_valid[:, 0], XYZ_valid[:, 1], XYZ_valid[:, 2])
# ax.scatter(XYZ_valid[0, 0], XYZ_valid[0, 1], XYZ_valid[0, 2], s=100, c='red')
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# plt.title('3D points after outlier removal')
# plt.show()



'''
plot outputs
'''
# plt.plot(output['GT_depths'])
# plt.plot(output['depths'])
# plt.title('depth estimation')
# plt.ylabel('CPR compression depth in mm')
# plt.show()

# plt.plot(output['interpolated_norm'])
# plt.plot(output['peaks'], output['interpolated_norm'][output['peaks']], "x")
# plt.plot(output['valleys'], output['interpolated_norm'][output['valleys']], "o")
# plt.title('peaks and valleys')
# plt.show()

# plt.plot(output['kinect_inter_depth_list'])
# plt.title('GT depth interpolated to kinect timestamps')
# plt.ylabel('CPR compression depth in mm')
# plt.show()






























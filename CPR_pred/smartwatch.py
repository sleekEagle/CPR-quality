import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

def interpolate_array(array,ts,target_ts):
    _,n=array.shape
    interp_list=[]
    for i in range(n):
        target_ts = np.squeeze(target_ts)
        # interp=utils.interpolate_between_ts(array[:,i],ts,target_ts,fit_window=5,deg=2)
        out_ts,valid_out_ts,interp=utils.interpolate_between_ts_cube(array[:,i],ts,target_ts,plot=False)
        interp_list.append(interp)
    interp_list=np.array(interp_list).T
    return interp_list,out_ts[valid_out_ts]

def extract_smartwatch_data(data_path,conf):
    #Android sensor codes
    TYPE_ACCELEROMETER=conf.smartwatch.TYPE_ACCELEROMETER
    TYPE_GYROSCOPE=conf.smartwatch.TYPE_GYROSCOPE
    TYPE_GRAVITY=conf.smartwatch.TYPE_GRAVITY
    TARGET_FREQ=conf.smartwatch.TARGET_FREQ

    with open(data_path) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        acc_ts,gyr_ts,grav_ts=[],[],[]
        acc_list,gyr_list,grav_list=[],[],[]
        acc_ts_machine,gyr_ts_machine,grav_ts_machine=[],[],[]

        for line in lines:
            sensor=int(line.split(',')[0])
            ts=int(line.split(',')[2])
            data_str=line.split('[')[1].split(']')[0].split(',')
            if len(data_str)<3:
                continue
            data=[float(value) for value in data_str]
            if sensor==TYPE_ACCELEROMETER:
                acc_ts.append(ts)
                acc_list.append(data)
                acc_ts_machine.append(line.split(',')[-1])
            elif sensor==TYPE_GYROSCOPE:
                gyr_ts.append(ts)
                gyr_list.append(data)
                gyr_ts_machine.append(line.split(',')[-1])
            elif sensor==TYPE_GRAVITY:
                grav_ts.append(ts)
                grav_list.append(data)
                grav_ts_machine.append(line.split(',')[-1])
            else:
                print('Unknown sensor type: ', sensor)
        
        acc_ts_machine=utils.get_ts_list(acc_ts_machine)
        gyr_ts_machine=utils.get_ts_list(gyr_ts_machine)
        grav_ts_machine=utils.get_ts_list(grav_ts_machine)

        acc_ts=np.array(acc_ts)
        acc_ts=acc_ts-acc_ts[0]
        gyr_ts=np.array(gyr_ts)
        gyr_ts=gyr_ts-gyr_ts[0]
        grav_ts=np.array(grav_ts)
        grav_ts=grav_ts-grav_ts[0]

        acc_ts_new=acc_ts_machine[0]+acc_ts*1e-9
        gyr_ts_new=gyr_ts_machine[0]+gyr_ts*1e-9
        grav_ts_new=grav_ts_machine[0]+grav_ts*1e-9

        #create target timestamps which we need to interpolate to
        start_ts=max(acc_ts_new[0],gyr_ts_new[0],grav_ts_new[0])
        end_ts=min(acc_ts_new[-1],gyr_ts_new[-1],grav_ts_new[-1])

        #interpolate the data to the target timestamps
        acc_list=np.array(acc_list)
        gyr_list=np.array(gyr_list)
        grav_list=np.array(grav_list)

        current_freq=len(acc_ts_new)/(acc_ts_new[-1]-acc_ts_new[0])
        target_ts=np.arange(start_ts,end_ts,1/TARGET_FREQ)
        target_ts = np.expand_dims(target_ts, axis=1)

        acc_interp,_=interpolate_array(acc_list,acc_ts_new,target_ts)
        acc_interp = np.squeeze(acc_interp)
        gyr_interp,_=interpolate_array(gyr_list,gyr_ts_new,target_ts)
        gyr_interp = np.squeeze(gyr_interp)
        grav_interp,valid_out_ts=interpolate_array(grav_list,grav_ts_new,target_ts)
        grav_interp = np.squeeze(grav_interp)

        output={}
        output['acc_interp']=acc_interp
        output['gyr_interp']=gyr_interp
        output['grav_interp']=grav_interp
        output['ts']=valid_out_ts
        return output,current_freq

data_path='D:\CPR_extracted\P1\s_5\smartwatch\smartwatch.txt'
gt_path=r'D:\CPR_extracted\P1\s_5'

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(conf : DictConfig) -> None:
    print(OmegaConf.to_yaml(conf))
    output,original_freq=extract_smartwatch_data(data_path,conf)

    # detect peaks and valleys
    mag=np.sqrt(np.square(output['acc_interp']).sum(axis=1))
    idx=np.arange(len(mag))/len(mag)
    #interpolate
    mag_interp=utils.interpolate_between_ts(mag,idx,idx,fit_window=30,deg=2)

    #detect peaks
    peaks,valleys=utils.find_peaks_and_valleys(mag_interp,distance=20,plot=True)

    #read GT depth sensor data
    depth_vals=np.array(utils.read_allnum_lines(os.path.join(gt_path,'depth_sensor.txt')))
    depth_ts=np.array(utils.read_allnum_lines(os.path.join(gt_path,'depth_sensor_ts.txt')))
    ts=output['ts']
    valid = (depth_ts > ts[0]) & (depth_ts < ts[-1])
    depth_ts=depth_ts[valid]
    depth_vals=depth_vals[valid]


    

    plt.plot(depth_vals)
    plt.plot(mag_interp)
    plt.show()







    pass

if __name__ == "__main__":
    main()

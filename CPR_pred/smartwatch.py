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

data_path='D:\CPR_extracted\P1\s_4\smartwatch\smartwatch.txt'
gt_path=r'D:\CPR_extracted\P1\s_4' 


def moving_normalize(signal, window_size):
    # Initialize the normalized signal with zeros
    normalized_signal = np.zeros(signal.shape)
    
    # Calculate the half window size for indexing
    half_window = window_size // 2
    
    for i in range(len(signal)):
        # Determine the start and end of the window
        start = max(i - half_window, 0)
        end = min(i + half_window + 1, len(signal))
        
        # Calculate local mean and standard deviation
        local_mean = np.mean(signal[start:end])
        local_std = np.std(signal[start:end])
        
        # Normalize the current value
        if local_std > 0:  # Avoid division by zero
            normalized_signal[i] = (signal[i] - local_mean) / local_std
        else:
            normalized_signal[i] = signal[i] - local_mean
    
    return normalized_signal

root_dir=r'D:\CPR_extracted'

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(conf : DictConfig) -> None:
    print(OmegaConf.to_yaml(conf))

    subj_dirs=utils.get_dirs_with_str(root_dir, 'P')
    for subj_dir in subj_dirs:
        session_dirs=utils.get_dirs_with_str(subj_dir,'s')
        for session_dir in session_dirs:
            # print(f'Processing {session_dir}')
            # if session_dir!=r'D:\CPR_extracted\P12\s_11':
            #     continue
            data_path=os.path.join(session_dir,'smartwatch','smartwatch.txt')
            if not os.path.exists(data_path):
                print(f'{data_path} does not exist. Continuing...')
                continue
            output,original_freq=extract_smartwatch_data(data_path,conf)

            # detect peaks and valleys
            prominant_axis=np.argmax(np.std(output['acc_interp'],axis=0))
            mag=np.sqrt(np.square(output['acc_interp'][:,prominant_axis]))
            idx=np.arange(len(mag))/len(mag)
            #interpolate
            mag_interp=utils.interpolate_between_ts(mag,idx,idx,fit_window=100,deg=2)
            # plt.plot(mag)
            # plt.plot(mag_interp)

            #detect peaks
            pred_peaks,pred_valleys=utils.find_peaks_and_valleys(mag_interp,distance=30,plot=False)
            #get CPR peaks and vallyes
            # cpr_peaks=[int((peaks[peaks>idx][0]+idx)*0.5) for idx in valleys if idx<=peaks[-1]]
            # cpr_vallyes=[int((valleys[valleys>idx][0]+idx)*0.5) for idx in peaks if idx<=valleys[-1]]

            #calc frequency
            t=(output['ts'][-1]-output['ts'][0])/60
            cpr_freq_est=len(pred_peaks)/t

            # plt.plot(mag_interp)
            # plt.plot(cpr_peaks, mag_interp[cpr_peaks], "x")
            # plt.plot(cpr_vallyes, mag_interp[cpr_vallyes], "o")
            # plt.show()

            if conf.smartwatch.eval:
                #read GT depth sensor data
                depth_vals=np.array(utils.read_allnum_lines(os.path.join(session_dir,'depth_sensor.txt')))
                depth_ts=np.array(utils.read_allnum_lines(os.path.join(session_dir,'depth_sensor_ts.txt')))
                ts=output['ts']
                valid = (depth_ts > ts[0]) & (depth_ts < ts[-1])
                depth_ts=depth_ts[valid]
                depth_vals=depth_vals[valid]
                if conf.smartwatch.plot_data:
                    plt.plot(depth_ts,depth_vals)
                    plt.plot(ts,mag_interp)
                    plt.plot(ts[pred_peaks], mag_interp[pred_peaks], "x")
                    plt.plot(ts[pred_valleys], mag_interp[pred_valleys], "x")
                    plt.show()

                #evaluate
                idx=np.arange(len(depth_vals))/len(depth_vals)
                if np.isnan(depth_vals).any():
                    print(f'Nan values in depth sensor data. Skipping...')
                    continue
                depth_vals_interp=utils.interpolate_between_ts(depth_vals,idx,idx,fit_window=20,deg=2)
                #running normalize 
                window_size = 100
                depth_vals_norm=moving_normalize(depth_vals_interp, window_size)
                # plt.plot(depth_vals_interp)
                # plt.plot(depth_vals_norm)

                GT_peaks,GT_valleys=utils.find_peaks_and_valleys(depth_vals_norm,distance=7,plot=conf.smartwatch.plot_data)

                # plt.plot(depth_vals)
                plt.plot(depth_vals_norm)
                plt.plot(GT_peaks,depth_vals_norm[GT_peaks], "x")
                plt.plot(GT_valleys,depth_vals_norm[GT_valleys], "x")
                plt.show()

                #calc freq
                t=(depth_ts[-1]-depth_ts[0])/60
                cpr_freq_GT=len(GT_peaks)/t

                freq_error=abs(cpr_freq_GT-cpr_freq_est)
                print(f'session_dir: {session_dir}. fequency error: {freq_error:.2f} Hz')

                # print(f'Estimated CPR frequency: {cpr_freq_est:.2f} Hz. GT CPR frequency: {cpr_freq_GT:.2f} Hz')

                # GT_peak_ts,GT_valleys_ts=depth_ts[GT_peaks],depth_ts[GT_valleys]
                # pred_peak_ts,pred_valleys_ts=output['ts'][cpr_peaks],output['ts'][cpr_vallyes]

                # peak_error=np.mean([min(abs(GT_peak_ts-pred)) for pred in pred_peak_ts])
                # valley_error=np.mean([min(abs(GT_valleys_ts-pred)) for pred in pred_valleys_ts])
                # avg_error=(peak_error+valley_error)*0.5*1000
                # print(f'average salient point error : {avg_error:.2f} ms')
    



if __name__ == "__main__":
    main()

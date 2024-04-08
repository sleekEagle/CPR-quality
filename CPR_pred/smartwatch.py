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
    
root_dir=r'D:\CPR_extracted'

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def create_dataset(conf : DictConfig) -> None:
    print(OmegaConf.to_yaml(conf))

    subj_dirs=utils.get_dirs_with_str(root_dir, 'P')
    gt_data,sw_data=[],[]
    depth_data,c_comp_data=[],[]
    part_data=[]

    for subj_dir in subj_dirs:
        session_dirs=utils.get_dirs_with_str(subj_dir,'s')
        for session_dir in session_dirs:
            # if session_dir!=r'D:\CPR_extracted\P13\s_8':
            #     continue
            print(f'Processing {session_dir}')
            data_path=os.path.join(session_dir,'smartwatch','smartwatch.txt')
            if not os.path.exists(data_path):
                print(f'{data_path} does not exist. Continuing...')
                continue
            output,original_freq=extract_smartwatch_data(data_path,conf)
            #read GT depth sensor data
            depth_vals=np.array(utils.read_allnum_lines(os.path.join(session_dir,'depth_sensor.txt')))
            #read GT depth sensor data
            if np.isnan(depth_vals).any():
                print(f'Nan values in depth sensor data. Skipping...')
                continue
            depth_ts=np.array(utils.read_allnum_lines(os.path.join(session_dir,'depth_sensor_ts.txt')))

            sw_ts=output['ts']
            t=(depth_ts[-1]-depth_ts[0])/60
            valid = (depth_ts > sw_ts[0]) & (depth_ts < sw_ts[-1])
            depth_ts=depth_ts[valid]
            depth_vals=depth_vals[valid]

            t=(depth_ts[-1]-depth_ts[0])/60
            #get number of zero crossings
            depth_vals_norm_=utils.moving_normalize(depth_vals, 100)
            num_zero_crossings = len(np.where(np.diff(np.sign(depth_vals_norm_)))[0])/t
            fit_window=int(1/num_zero_crossings*2700)
            idx=np.arange(len(depth_vals))/len(depth_vals)
            depth_vals_interp=utils.interpolate_between_ts(depth_vals,idx,idx,fit_window=fit_window,deg=2)
            depth_vals_norm=utils.moving_normalize(depth_vals_interp, 100)
            dist=int(1/num_zero_crossings*1000)
            GT_peaks,GT_valleys,idx=utils.find_peaks_and_valleys(depth_vals_norm,distance=dist,plot=False)

            #create windows
            window_len=conf.smartwatch.rate_window*conf.smartwatch.TARGET_FREQ
            increment=int(window_len*conf.smartwatch.overlap)
            idx=0

            while(((idx+window_len)<len(sw_ts)) and ((idx+window_len)<len(depth_ts)) and (sw_ts[-1]>sw_ts[idx+window_len]) and (depth_ts[-1]>depth_ts[idx+window_len])):
                gt_window=depth_vals[idx:idx+window_len]
                sw_window=np.array([output['acc_interp'][idx:idx+window_len],
                           output['acc_interp'][idx:idx+window_len],
                           output['acc_interp'][idx:idx+window_len]])
                GT_peaks_window=GT_peaks[(GT_peaks>=idx) & (GT_peaks<(idx+window_len))]-idx
                GT_valleys_window=GT_valleys[(GT_valleys>=idx) & (GT_valleys<(idx+window_len))]-idx
                idx+=increment
                # plt.plot(gt_window)
                # plt.scatter(GT_peaks_window,gt_window[GT_peaks_window],c='r')
                # plt.scatter(GT_valleys_window,gt_window[GT_valleys_window],c='g')
                all_pts=np.concatenate((GT_peaks_window,GT_valleys_window))
                all_pts.sort()
                n,d=0,0
                for pt in GT_peaks_window:
                    i=np.argwhere(all_pts==pt)[0][0]
                    if ((i+1)<len(all_pts)) and (all_pts[i+1] in GT_valleys_window):
                        d+=abs(gt_window[pt]-gt_window[all_pts[i+1]])
                        n+=1
                    if ((i-1)>=0) and (all_pts[i-1] in GT_valleys_window):
                        d+=abs(gt_window[pt]-gt_window[all_pts[i-1]])
                        n+=1
                depth=d/n
                n_cmp=0.5*(len(GT_peaks_window)+len(GT_valleys_window))
                part = int(os.path.basename(os.path.dirname(session_dir))[1:])

                gt_data.append(gt_window)
                sw_data.append(sw_window)
                depth_data.append(depth)
                c_comp_data.append(n_cmp)
                part_data.append(part)
    
    #write data to file
    gt_data=np.array(gt_data)
    sw_data=np.array(sw_data)
    depth_data=np.array(depth_data)
    c_comp_data=np.array(c_comp_data)
    part_data=np.array(part_data)

    out_dir=os.path.join(root_dir,'smartwatch_dataset')
    os.makedirs(out_dir,exist_ok=True)
    np.save(os.path.join(out_dir,'gt_data'), gt_data)
    np.save(os.path.join(out_dir,'sw_data'), sw_data)
    np.save(os.path.join(out_dir,'depth_data'), depth_data)
    np.save(os.path.join(out_dir,'c_comp_data'), c_comp_data)
    np.save(os.path.join(out_dir,'part_data'), part_data)

    np.load(os.path.join(out_dir,'gt_data.npy'))


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def detect_peaks(conf : DictConfig) -> None:
    print(OmegaConf.to_yaml(conf))

    subj_dirs=utils.get_dirs_with_str(root_dir, 'P')
    for subj_dir in subj_dirs:
        session_dirs=utils.get_dirs_with_str(subj_dir,'s')
        for session_dir in session_dirs:
            # if session_dir!=r'D:\CPR_extracted\P13\s_8':
            #     continue
            print(f'Processing {session_dir}')
            data_path=os.path.join(session_dir,'smartwatch','smartwatch.txt')
            if not os.path.exists(data_path):
                print(f'{data_path} does not exist. Continuing...')
                continue
            output,original_freq=extract_smartwatch_data(data_path,conf)
            #read GT depth sensor data
            depth_vals=np.array(utils.read_allnum_lines(os.path.join(session_dir,'depth_sensor.txt')))
            #read GT depth sensor data
            if np.isnan(depth_vals).any():
                print(f'Nan values in depth sensor data. Skipping...')
                continue
            depth_ts=np.array(utils.read_allnum_lines(os.path.join(session_dir,'depth_sensor_ts.txt')))
            t=(depth_ts[-1]-depth_ts[0])/60
            #get number of zero crossings
            depth_vals_norm_=utils.moving_normalize(depth_vals, 100)
            num_zero_crossings = len(np.where(np.diff(np.sign(depth_vals_norm_)))[0])/t
            fit_window=int(1/num_zero_crossings*2700)
            idx=np.arange(len(depth_vals))/len(depth_vals)
            depth_vals_interp=utils.interpolate_between_ts(depth_vals,idx,idx,fit_window=fit_window,deg=2)
            depth_vals_norm=utils.moving_normalize(depth_vals_interp, 100)
            dist=int(1/num_zero_crossings*1000)
            GT_peaks,GT_valleys,idx=utils.find_peaks_and_valleys(depth_vals_norm,distance=dist,plot=False)
            # plt.plot(depth_vals)
            # plt.plot(GT_peaks,depth_vals[GT_peaks], "x")
            # plt.plot(GT_valleys,depth_vals[GT_valleys], "x")
            # plt.show()
            out_file=os.path.join(session_dir,'smartwatch','depth_sensor_peaks.txt')
            with open(out_file, 'w') as f:
                for item in GT_peaks:
                    f.write("%s\n" % item)
            out_file=os.path.join(session_dir,'smartwatch','depth_sensor_valleys.txt')
            with open(out_file, 'w') as f:
                for item in GT_valleys:
                    f.write("%s\n" % item)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(conf : DictConfig) -> None:
    print(OmegaConf.to_yaml(conf))

    subj_dirs=utils.get_dirs_with_str(root_dir, 'P')
    for subj_dir in subj_dirs:
        session_dirs=utils.get_dirs_with_str(subj_dir,'s')
        for session_dir in session_dirs:
            # if session_dir!=r'D:\CPR_extracted\P1\s_2':
            #     continue
            data_path=os.path.join(session_dir,'smartwatch','smartwatch.txt')
            if not os.path.exists(data_path):
                print(f'{data_path} does not exist. Continuing...')
                continue
            output,original_freq=extract_smartwatch_data(data_path,conf)
            t=(output['ts'][-1]-output['ts'][0])
            if t<conf.smartwatch.rate_window:
                print('session too short. Skipping...')
                continue

            prominant_axis=np.argmax(np.std(output['gyr_interp'],axis=0))
            mag=output['gyr_interp'][:,prominant_axis]

            window_len=conf.smartwatch.rate_window*conf.smartwatch.TARGET_FREQ
            dom_freq_est_list=[]
            for j,i in enumerate(range(0,len(mag)-window_len,window_len//3)):
                window=mag[i:i+window_len]
                dom_freq_est=utils.get_dominant_freq(window,conf.smartwatch.TARGET_FREQ)
                dom_freq_est_list.append(dom_freq_est)
            

                

            # fft_values = np.abs(np.fft.fft(mag))
            # freq_bins = np.fft.fftfreq(len(mag), 1/60)
            # # freq_idx=np.argwhere(freq_bins>=0)
            # # freq_bins=freq_bins[freq_idx]
            # # fft_values=fft_values[freq_idx]
            
            # valid_cpr_freq=np.array([10,300])/60
            # valid_idx=np.argwhere((freq_bins>=valid_cpr_freq[0]) & (freq_bins<=valid_cpr_freq[1]))[:,0]
            # fft_values=fft_values[valid_idx]
            # freq_bins=freq_bins[valid_idx]
            # peak,_,_=utils.find_peaks_and_valleys(fft_values,distance=len(fft_values),plot=False)
            # dom_freq_est=freq_bins[peak]
            # plt.plot(fft_values)
            # plt.title('smartwatch')
            # plt.show()

            # plt.plot(output['ts'],mag)
            # plt.show()

            # # detect peaks and valleys
            # prominant_axis=np.argmax(np.std(output['gyr_interp'],axis=0))
            # mag=output['gyr_interp'][:,prominant_axis]
            
            # # mag_high = np.percentile(np.abs(mag), 90)
            # # mag=np.sqrt(np.sum(np.square(output['acc_interp']),axis=1))
            # # mag=np.sqrt(np.square(output['acc_interp'][:,prominant_axis]))
            # idx=np.arange(len(mag))/len(mag)

            # mag_norm_=utils.moving_normalize(mag,100)
            # num_zero_crossings = len(np.where(np.diff(np.sign(mag_norm_)))[0])/t
            # #interpolate
            # if num_zero_crossings<500:
            #     fit_window=int(8000/num_zero_crossings)
            # else:
            #     fit_window=int(12000/num_zero_crossings)
            # # print(f'fit_window: {fit_window}')'
            # mag_interp=utils.interpolate_between_ts(mag,idx,idx,fit_window=fit_window,deg=2)
            # # plt.plot(mag_norm_)   
            # # plt.plot(mag_interp)
            # # plt.show()
            # mag_norm=utils.moving_normalize(mag_interp,100)
            # # plt.plot(mag)
            # # plt.plot(output['ts'],mag_interp)
            # # plt.plot(depth_ts,depth_vals_norm)

            # #detect peaks
            # dist=int(1/num_zero_crossings*8500)
            # # if abs(num_zero_crossings-174.999)<1:
            # #     print('here')
            # # else:
            # #     continue
            # if num_zero_crossings<500:
            #     dist=4900/num_zero_crossings
            # else:
            #     dist=15000/num_zero_crossings
            # # if num_zero_crossings<110:
            # #     dist=45
            # # elif num_zero_crossings<150:
            # #     dist=40
            # # elif num_zero_crossings<200:
            # #     dist=35
            # # elif num_zero_crossings<250:
            # #     dist=30
            # # elif num_zero_crossings<300:
            # #     dist=20
            # # elif num_zero_crossings<350:
            # #     dist=15
            # # elif num_zero_crossings<400:
            # #     dist=10
            # # else:
            # #     continue

            # print(f'Number of zero crossings: {num_zero_crossings}')
            # pred_peaks,pred_valleys,idx=utils.find_peaks_and_valleys(mag_norm,distance=dist,prominence=1,plot=False)
            # pred_peaks=[pred_peaks,pred_valleys][idx]
            # #get CPR peaks and vallyes
            # # cpr_peaks=[int((peaks[peaks>idx][0]+idx)*0.5) for idx in valleys if idx<=peaks[-1]]
            # # cpr_vallyes=[int((valleys[valleys>idx][0]+idx)*0.5) for idx in peaks if idx<=valleys[-1]]

            # #calc frequency
            # cpr_freq_est=len(pred_peaks)/t

            # # plt.plot(mag)
            # # plt.plot(pred_peaks, mag[pred_peaks], "x")
            # # plt.plot(pred_valleys, mag[pred_valleys], "o")
            # # plt.show()


            if conf.smartwatch.eval:
                depth_vals=np.array(utils.read_allnum_lines(os.path.join(session_dir,'depth_sensor.txt')))
                #read GT depth sensor data
                if np.isnan(depth_vals).any():
                    print(f'Nan values in depth sensor data. Skipping...')
                    continue
                depth_ts=np.array(utils.read_allnum_lines(os.path.join(session_dir,'depth_sensor_ts.txt')))
                ts=output['ts']
                t=(depth_ts[-1]-depth_ts[0])/60
                valid = (depth_ts > ts[0]) & (depth_ts < ts[-1])
                depth_ts=depth_ts[valid]
                depth_vals=depth_vals[valid]
                # print(f't: {t}')

                sampling_rate=len(depth_ts)/(depth_ts[-1]-depth_ts[0])

                window_len=int(conf.smartwatch.rate_window*sampling_rate)
                dom_freq_list=[]
                for i in range(0,len(depth_vals)-window_len,window_len//3):
                    window=depth_vals[i:i+window_len]
                    dom_freq=utils.get_dominant_freq(window,sampling_rate)
                    dom_freq_list.append(dom_freq)

                min_len=min(len(dom_freq_list),len(dom_freq_est_list))
                error=np.mean(np.abs(np.array(dom_freq_est_list[0:min_len])-np.array(dom_freq_list[0:min_len])))
                print(f'{session_dir} error: {error*5} Hz')
                # print(f'{session_dir} error: {np.mean(dom_freq_list)-np.mean(dom_freq_est_list)} Hz')


                # fft_values = np.abs(np.fft.fft(depth_vals))
                # freq_bins = np.fft.fftfreq(len(depth_vals), 1/sampling_rate)
                # valid_idx=np.argwhere((freq_bins>=valid_cpr_freq[0]) & (freq_bins<=valid_cpr_freq[1]))[:,0]
                # fft_values=fft_values[valid_idx]
                # freq_bins=freq_bins[valid_idx]
                # peak,_,_=utils.find_peaks_and_valleys(fft_values,distance=len(fft_values),plot=False)
                # dom_freq_GT=freq_bins[peak]
                # print(f'{session_dir} Difference: {60*(dom_freq_est-dom_freq_GT)} Hz')

                # plt.plot(fft_values)
                # plt.title('Depth sensor FFT')
                # plt.show()

                #evaluate
                # idx=np.arange(len(depth_vals))/len(depth_vals)
                # if np.isnan(depth_vals).any():
                #     print(f'Nan values in depth sensor data. Skipping...')
                #     continue
                # #get number of zero crossings
                # depth_vals_norm_=utils.moving_normalize(depth_vals, 100)
                # num_zero_crossings = len(np.where(np.diff(np.sign(depth_vals_norm_)))[0])/t

                # #select fit window
                # fit_window=int(1/num_zero_crossings*2700)
                # depth_vals_interp=utils.interpolate_between_ts(depth_vals,idx,idx,fit_window=fit_window,deg=2)
                # # plt.plot(depth_vals)
                # # plt.plot(depth_vals_interp)
                # # plt.show()
                # #running normalize 
                # depth_vals_norm=utils.moving_normalize(depth_vals_interp, 100)
                # # plt.plot(depth_vals_interp)
                # # plt.plot(depth_vals_norm)

                # GT_peaks,GT_valleys,idx=utils.find_peaks_and_valleys(depth_vals_norm,distance=7,plot=False)

                # # plt.plot(depth_vals)
                # # # plt.plot(depth_vals_norm)
                # # plt.plot(GT_peaks,depth_vals[GT_peaks], "x")
                # # plt.plot(GT_valleys,depth_vals[GT_valleys], "x")
                # # plt.show()

                # num_compressions=len(GT_peaks)
                # if num_compressions<10:
                #     print('session too short. Skipping...')
                #     continue

                # #calc freq
                # cpr_freq_GT=len(GT_peaks)/t

                # freq_error=abs(cpr_freq_GT-cpr_freq_est)
                # # print(f'session_dir: {session_dir}. fequency error: {freq_error:.2f} Hz')

                # # print(f'{session_dir} Estimated CPR frequency: {cpr_freq_est:.2f} Hz. GT CPR frequency: {cpr_freq_GT:.2f} Hz. Error : {freq_error:.2f} Hz.')
                # print(f'{session_dir} Difference: {cpr_freq_est-cpr_freq_GT}')

                # # GT_peak_ts,GT_valleys_ts=depth_ts[GT_peaks],depth_ts[GT_valleys]
                # # pred_peak_ts,pred_valleys_ts=output['ts'][cpr_peaks],output['ts'][cpr_vallyes]

                # # peak_error=np.mean([min(abs(GT_peak_ts-pred)) for pred in pred_peak_ts])
                # # valley_error=np.mean([min(abs(GT_valleys_ts-pred)) for pred in pred_valleys_ts])
                # # avg_error=(peak_error+valley_error)*0.5*1000
                # # print(f'average salient point error : {avg_error:.2f} ms')
    



if __name__ == "__main__":
    create_dataset()

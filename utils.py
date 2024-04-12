from datetime import datetime
import os
import shutil
import numpy as np

def get_XYZ(x,y,depth):
    k=np.array([[615.873673811006,0,640.803032851225],[0,615.918359977960,365.547839233105],[0,0,1]])
    X=(x-k[0,2])*depth/k[0,0]
    Y=(y-k[1,2])*depth/k[1,1]
    Z=depth
    return X,Y,Z

def get_float_time(time_object):
    time_float = float(time_object.hour * 3600 
    + time_object.minute * 60 
    + time_object.second
    + time_object.microsecond*1e-6)
    return time_float

def get_ts_list(lines):
    ts_lines=[get_float_time(datetime.strptime(line.strip(), "%H_%M_%S.%f").time()) for line in lines]
    return ts_lines

def get_kinect_ts_list(kin_ts_path):
    with open(kin_ts_path, 'r') as file:
        # Read all lines from the file into a list
        lines = file.readlines()
        kinect_ts_lines = get_ts_list(lines)
    return kinect_ts_lines


'''
file operations
'''

def get_dirs_with_str(path,str,i=0,j=0):
    directories = [os.path.join(path,name) for name in os.listdir(path) if (os.path.isdir(os.path.join(path, name)) and (str in name[i:len(name)-j]))]
    return directories
def get_files_with_str(path,str,i=0,j=0):
    files = [os.path.join(path,file) for file in os.listdir(path) if str in file[i:len(file)-j]]
    return files

'''
move source dir or file into the target dir
if source is a file move the file into the target dir
if source is a dir move the dir into a sub directory in target dir
'''
def move_into_dir(source,target_dir,copy=True):
    if not os.path.exists(target_dir):
            os.makedirs(target_dir)

    if not os.path.exists(source):
        raise Exception(f"{source} does not exist")
    
    if os.path.isfile(source):
        shutil.move(source, target_dir)
        return 0
    dirname=os.path.basename(source)
    target_sub_dir=os.path.join(target_dir, dirname)
    if os.path.exists(target_sub_dir):
        shutil.rmtree(target_sub_dir)
    if copy:
        shutil.copytree(source, target_sub_dir)
    else:
        shutil.move(source, target_sub_dir)
    return 0

def list_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirectories

def list_files(directory,ext):
    files = [f for f in os.listdir(directory) if (os.path.isfile(os.path.join(directory, f)) and (f.split('.')[-1].lower()==ext))]
    return files

def copy_files(file_list, source_directory, destination_directory):
    try:
        # Create the destination directory if it doesn't exist
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)

        # Copy each file from the source to the destination directory
        for file_name in file_list:
            source_path = os.path.join(source_directory, file_name)
            destination_path = os.path.join(destination_directory, file_name)
            shutil.copy2(source_path, destination_path)  # Using shutil.copy2 to preserve metadata

        print("Files copied successfully.")
    except FileNotFoundError:
        print("Source directory not found.")
    except FileExistsError:
        print("Destination directory already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_allnum_lines(path):
    with open(path, 'r') as file:
        # Read all lines into a list
        all_lines = file.readlines()
    values=[]
    for line in all_lines:
        try:
            f=float(line.strip())
            values.append(f)
        except:
            pass
    return values


'''
in_values: values in the input signal
in_ts: timestamps of the input signal
out_ts: the timestamps where the input signal is interpolated to

in_values and in_ts must have the same dimentions
out_ts must lie within in_ts. There should be no values of out_ts outside in_ts
'''
def interpolate_between_ts(in_values,in_ts,out_ts,fit_window=8,deg=4):
    #select the relavent kinect data section based on depth sensor data
    t_start,t_end=in_ts[0],in_ts[-1]
    out_ts_start,out_ts_end=out_ts[0],out_ts[-1]
    assert out_ts_start>=t_start and out_ts_end<=t_end, "out ts are outside of depth sensor. Quitting..."
    # fit a polyormial and interpolate the GT depth at kinect ts values
    out_interp=np.zeros_like(out_ts)
    num_data=np.zeros_like(out_ts)

    for i in range(len(out_ts)-fit_window+1):
        out_ts_vals=out_ts[i:i+fit_window]
        sensor_start_arg=np.min(np.argwhere(in_ts>=out_ts_vals[0]))
        sensor_end_arg=np.max(np.argwhere(in_ts<=out_ts_vals[-1]))
        if sensor_start_arg>=sensor_end_arg:
            pred_depth=in_values[sensor_start_arg]*np.ones((fit_window,1))
        else:
            fit_data_ts=in_ts[sensor_start_arg:sensor_end_arg+1]
            fit_data_depth=in_values[sensor_start_arg:sensor_end_arg+1]
            fit_data_ts=fit_data_ts-in_ts[sensor_start_arg]
            out_ts_vals=out_ts_vals-in_ts[sensor_start_arg]
            
            p=np.polyfit(fit_data_ts,fit_data_depth,deg=deg)
            pred_depth=np.polyval(p,out_ts_vals)

            # plt.plot(fit_data_ts,fit_data_depth)
            # plt.plot(out_ts_vals,pred_depth)
        
        out_interp[i:i+fit_window]+=np.squeeze(pred_depth)
        num_data[i:i+fit_window]+=1

    where=np.argwhere(num_data>0)
    out_interp[where]/=num_data[where]

    # import matplotlib.pyplot as plt
    # plt.plot(in_ts,in_values)
    # plt.plot(out_ts,out_interp)
    # plt.show()

    return out_interp

def interpolate_between_ts_cube(signal,in_ts,out_ts,plot=False):
    from scipy.interpolate import CubicSpline
    start_ts,end_ts=in_ts[0],in_ts[-1]
    valid_out_ts= (out_ts>=start_ts)&(out_ts<=end_ts)
    out_ts=out_ts[valid_out_ts]
    in_ts_norm=in_ts-start_ts
    out_ts_norm=out_ts-start_ts
    #remove duplicate or wrong timestamps
    valid_idx=np.diff(in_ts_norm)>0
    if in_ts_norm[-1]>in_ts_norm[-2]:
        valid_idx=np.concatenate((valid_idx,[True]))
    spl=CubicSpline(in_ts_norm[valid_idx],signal[valid_idx])
    pred=spl(out_ts_norm)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(in_ts,signal)
        plt.plot(out_ts,pred)
        plt.show()
    return out_ts,valid_out_ts,pred

# animate_pt_seq(np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3]]))
def animate_pt_seq(data, interval=0.5):
    import matplotlib.pyplot as plt
    # data=np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3]])
    fig=plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    for i in range(len(data)):
        plt.cla()
        ax.set_xlim([np.min(data[:,0]), np.max(data[:,0])])
        ax.set_ylim([np.min(data[:,1]), np.max(data[:,1])])
        ax.set_zlim([np.min(data[:,2]), np.max(data[:,2])])
        ax.set_xlabel('X', fontweight ='bold')
        ax.set_ylabel('Y', fontweight ='bold')
        ax.set_zlabel('Z', fontweight ='bold')
        ax.scatter3D(data[i,0], data[i,1], data[i,2], color = "green")
        plt.pause(interval)
    plt.show()

'''
plot a list of points on an image
'''
def plot_points(image,points):
    import cv2
    # Draw circles at the specified points
    for point in points:
        cv2.circle(image, point, 5, (0, 255, 0), -1) 
    return image


def show_img(image,show_coords=False):
    import cv2
    window_name='image'
    # Define the mouse callback function
    if show_coords:
        def show_coordinates(event, x, y, flags, param):
            # Check for mouse movement
            if event == cv2.EVENT_MOUSEMOVE:
                # Update the window title to show the coordinates
                coordinates = f'Coordinates: X={x}, Y={y}'
                cv2.setWindowTitle(window_name, coordinates)

    # Check if the image was successfully loaded
    if image is not None:
        if show_coords:
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, show_coordinates)
        cv2.imshow(window_name, image)

        # Wait for a key press and then close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to load the image.")

def select_points(image):
    import cv2
    # Function to handle mouse clicks
    def click_event(event, x, y, flags, param):
        # Check if the event is a left mouse click
        if event == cv2.EVENT_LBUTTONDOWN:
            # Optionally, you can mark the clicked point and display it
            cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.imshow("image", image)
            global coordinates
            coordinates = (x, y)
            print(f'coordinates: {coordinates}')

    cv2.imshow("image", image)
    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return coordinates


#crop an image given a bounding box and padding
def crop_img_bb(img,hand_bb,pad):
    h,w,_=img.shape
    img_crop=img[max(0,hand_bb[1]-pad):min(h,hand_bb[3]+pad),max(0,hand_bb[0]-pad):min(w,hand_bb[2]+pad)]
    return img_crop


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

#find peaks and valleys in a 1D signal
#signal must be normalized in [-1,+1] in a moving manner
def find_peaks_and_valleys(signal, distance=10,height=0.2,prominence=(None, None),plot=False):
    from scipy.signal import find_peaks
    peaks, p_properties  = find_peaks(signal, distance=distance,height=height,prominence=prominence)
    valleys, v_properties = find_peaks(-signal, distance=distance,height=height,prominence=prominence)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(signal)
        plt.scatter(peaks, signal[peaks], c='r', label='Peaks')
        plt.scatter(valleys, signal[valleys], c='g', label='Valleys')
        plt.legend()
        plt.show()
    best_idx=-1
    if len(peaks)>1:
        best_idx=np.argmax([p_properties['prominences'].mean(),v_properties['prominences'].mean()])
    return peaks, valleys,best_idx

def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size)/window_size, mode='valid')

#get dominantt freq in Hz
def get_dominant_freq(signal,sample_freq):
    valid_cpr_freq=np.array([10,300])/60
    
    fft_values = np.abs(np.fft.fft(signal))
    freq_bins = np.fft.fftfreq(len(signal), 1/sample_freq)
    valid_idx=np.argwhere((freq_bins>=valid_cpr_freq[0]) & (freq_bins<=valid_cpr_freq[1]))[:,0]
    fft_values=fft_values[valid_idx]
    freq_bins=freq_bins[valid_idx]
    #get the highest frequency (except the DC component)
    dom_freq_est=freq_bins[1:][np.argmax(fft_values[1:])]

    return dom_freq_est



#************************************************************
#*********handle smartwatch data**************************
#************************************************************

def interpolate_array(array,ts,target_ts):
    _,n=array.shape
    interp_list=[]
    for i in range(n):
        target_ts = np.squeeze(target_ts)
        # interp=utils.interpolate_between_ts(array[:,i],ts,target_ts,fit_window=5,deg=2)
        out_ts,valid_out_ts,interp=interpolate_between_ts_cube(array[:,i],ts,target_ts,plot=False)
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
        
        acc_ts_machine=get_ts_list(acc_ts_machine)
        gyr_ts_machine=get_ts_list(gyr_ts_machine)
        grav_ts_machine=get_ts_list(grav_ts_machine)

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




from datetime import datetime
import os
import shutil
import numpy as np

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
    out_ts=out_ts[(out_ts>=start_ts)&(out_ts<=end_ts)]
    in_ts_norm=in_ts-start_ts
    out_ts_norm=out_ts-start_ts
    spl=CubicSpline(in_ts_norm,signal)
    pred=spl(out_ts_norm)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(in_ts,signal)
        plt.plot(out_ts,pred)
        plt.show()
    return out_ts,pred

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

def show_img(image):
    import cv2
    # Check if the image was successfully loaded
    if image is not None:
        # Display the image
        cv2.imshow('Image', image)

        # Wait for a key press and then close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to load the image.")

#crop an image given a bounding box and padding
def crop_img_bb(img,hand_bb,pad):
    h,w,_=img.shape
    img_crop=img[max(0,hand_bb[1]-pad):min(h,hand_bb[3]+pad),max(0,hand_bb[0]-pad):min(w,hand_bb[2]+pad)]
    return img_crop

#find peaks and valleys in a 1D signal
def find_peaks_and_valleys(signal, distance=10):
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(signal, distance=distance)
    valleys, _ = find_peaks(-signal, distance=distance)
    return peaks, valleys


#************************************************************
#*********get direcoties and files**************************
#************************************************************




from datetime import datetime
import os
import shutil

def get_float_time(time_object):
    time_float = float(time_object.hour * 3600 
    + time_object.minute * 60 
    + time_object.second
    + time_object.microsecond*1e-6)
    return time_float

def get_kinect_ts_list(kin_ts_path):
    with open(kin_ts_path, 'r') as file:
        # Read all lines from the file into a list
        lines = file.readlines()
        kinect_ts_lines=[get_float_time(datetime.strptime(line.strip(), "%H_%M_%S.%f").time()) for line in lines]
    return kinect_ts_lines

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

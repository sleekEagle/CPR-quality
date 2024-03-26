import time

while True:
    # Get the current time in epoch format
    epoch_time = int(time.time_ns())
    
    # Print the epoch time
    print(epoch_time,end='\r')
    
    # Wait for 1 second before printing the next time
    time.sleep(0.01)

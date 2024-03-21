import board 
import adafruit_mpu6050
import threading

def collect_data(duration, filename):
    i2c = board.I2C()  # uses board.SCL and board.SDA
    mpu = adafruit_mpu6050.MPU6050(i2c)
    
    # Construct the full path with the desired directory
    directory_path = os.path.join('..', 'imu_data')
    full_path = os.path.join(directory_path, filename)
    
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("imu_data directory is created")
    
    # Check if the file exists, delete it if it does
    if os.path.exists(full_path):
        os.remove(full_path)
        print(f"File {full_path} already existed. Overwriting...")
    
    # Open the file in binary write mode
    with open(filename, 'wb') as file:
        end_time = time.time() + duration
        
        def collect_and_store():
            if time.time() < end_time:
                # Simulating the collection of IMU data (6 values) and a timestamp
                ax = mpu.acceleration[0]
                ay = mpu.acceleration[1]
                az = mpu.acceleration[2]
                gx = mpu.gyro[0]
                gy = mpu.gyro[1]
                gz = mpu.gyro[2]

                imu_data = [ax ay az gx gy gz] # Placeholder for actual IMU data collection
                timestamp = time.time()
                data_to_store = struct.pack('d' * 7, timestamp, *imu_data)
                
                # Write the packed data to the file
                file.write(data_to_store)
                
                # Schedule the next collection in 0.02 seconds
                threading.Timer(0.02, collect_and_store).start()
        
        # Start the first data collection
        collect_and_store()

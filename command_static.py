import time
import subprocess
import sys
import os
import csv
import datetime
import argparse
import cv2
import threading
import platform  # Add this to detect OS
from utils.video_cap import capture_video

def execute_c_program(c_program_path, c_program_args):
    command = [c_program_path] + c_program_args
    
    # Execute the C program
    try:
        print("Executing C program...")
        result = subprocess.run(command, check=True)
        print("C program executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing C program: {e}")

def capture_frame_and_save(folder_path, image_name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera, exiting thread")
        sys.exit()
        return
    ret, frame = cap.read()
    cap.release()
    if ret:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        image_path = os.path.join(folder_path, image_name)
        cv2.imwrite(image_path, frame)
        print("Image saved successfully:", image_path)
    else:
        print("Error: Failed to capture frame")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser for radar params')
        
    parser.add_argument('-nf', '--nframes', type=int, default=100, help='Number of frames (default: 100)')
    parser.add_argument('-nc', '--nchirps', type=int, default=182, help='Number of chirps in a frame, usually 182')
    parser.add_argument('-tc', '--timechirp', type=int, default=72, help='Chirp time in microseconds, usually 72')
    parser.add_argument('-s', '--samples', type=int, default=256, help='Number of ADC samples, or range bins, usually 256')
    parser.add_argument('-r', '--rate', type=int, default=4400, help='Sampling rate, usually 4400')
    parser.add_argument('-tf', '--timeframe', type=int, default=150, help='Periodicity or Frame time in milliseconds (default: 150)')
    parser.add_argument('-l', '--length', type=int, default=-1, help='Initial length (default: -1)')
    parser.add_argument('-r0', '--radial', type=int, default=-1, help='Initial radial distance (default: -1)')
    
    parser.add_argument('-d', '--descp', type=str, help='Data description')
    parser.add_argument('-camera', action='store_true')
    parser.add_argument('-imu', action='store_true')
    parser.add_argument('-csv_store', action='store_true')
    
    # Detect operating system
    current_os = platform.system()
    
    # OS-specific commands for Linux and Windows
    if current_os == "Linux":
        os.system("sudo macchanger --mac=c0:18:50:da:37:e0 eth0")
        # os.system("sudo chmod a+rw /dev/ttyACM0")
    elif current_os == "Windows":
        print("Skipping macchanger command on Windows...")

    ans1 = input("Have you connected the ethernet? yes/no: ")
    camera_pass = False
    args = parser.parse_args()
    
    if args.camera:
        ans3 = input("Have you connected camera cable? yes/no: ")
        if ans3 == "yes":
            camera_pass = True
    elif not args.camera:
        camera_pass = True
    
    if ans1 == 'yes' and camera_pass:
        c_program_path = "/home/jetson/Desktop/BTP/data_collection/mmSnS/data_collect_mmwave_only_linux" if current_os == "Linux" else "C:\\Users\\acer\\Desktop\\mmSnS\\mmSnS\\data_collect_mmwave_only_win.exe"  # Use Windows path when needed
        
        image_folder_path = "./scene_annotation/" if current_os == "Linux" else ".\\scene_annotation\\"
        now = datetime.date.today()
        date_string = str(now.strftime('%Y-%m-%d'))
        n_frames = str(args.nframes)
        n_chirps = str(args.nchirps)
        tc = str(args.timechirp)
        adc_samples = str(args.samples)
        sampling_rate = str(args.rate)
        periodicity = str(args.timeframe)
        l = str(args.length)
        r0 = str(args.radial)
        descri = args.descp
        date_string += "_" + descri
        file_name = date_string + "_" + ".bin"
        image_name = date_string + "_" + ".jpg"
        c_program_args = [file_name, n_frames]
        
        if args.camera:
            capture_frame_and_save(image_folder_path, image_name)
        
        if args.imu:
            from utils.imu_data_collector import collect_data
            imu_duration = (int(n_frames) + 5) * int(periodicity) / 1000  # periodicity is in ms (collect for 5 extra frames)
            imu_filename = date_string + "_" + "_imu.bin"
            imu_thread = threading.Thread(target=collect_data, args=(imu_duration, imu_filename))
            imu_thread.start()
        
        execute_c_program(c_program_path, c_program_args)
        
        if args.imu:
            imu_thread.join()
        
        ans_to_keep = input('Do you want to keep the reading? yes/no : ')
        
        if ans_to_keep == 'no':
            os.remove(file_name)
            print(f"{file_name} deleted successfully")
            if args.imu:
                imu_file_path = os.path.join("./imu_data/", imu_filename)
                os.remove(imu_file_path)
                print(f"{imu_file_path} deleted successfully")
            sys.exit()
        
        if args.csv_store:
            bot_vel = float(input("Enter ground truth bot velocity in cm/s: "))
            expected_del_phi_peak = -(3.14 * bot_vel * 3 * 86 * 0.00001)
            file_path = "dataset.csv" if current_os == "Linux" else "dataset.csv"  # Use relative path for both systems
            
            data = [file_name, n_frames, n_chirps, tc, adc_samples, sampling_rate, periodicity, l, r0, descri, bot_vel, expected_del_phi_peak]
            if r0 == l:
                data.append('Straight')
            else:
                data.append('Oblique')
        
            with open(file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
                print('Data appended successfully')

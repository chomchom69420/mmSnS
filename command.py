import serial
import time
import subprocess
import sys
import os
import csv
import datetime
import argparse
def send_command_to_arduino(serial_port, command):
    try:
        # Initialize serial connection
        ser = serial.Serial(serial_port, 115200, timeout=1)

        # Wait for the serial connection to initialize
        time.sleep(2)

        # Send the command followed by a newline character
        ser.write((command + "\n").encode())

        print(f"'{command}' command sent to Arduino.")

        # Close the serial port
        ser.close()

    except serial.SerialException as e:
        print(f"Error: {e}")
        print("Could not open serial port. Make sure your Arduino is connected and the port is correct.")

def execute_c_program_and_control_arduino(arduino_port, c_program_path, c_program_args,pwm_value):
    # Send START command to Arduino
    send_command_to_arduino(arduino_port,"PWM"+pwm_value)
    time.sleep(1)
    send_command_to_arduino(arduino_port, "START"+pwm_value)
    time.sleep(1)
    command=[c_program_path] + c_program_args
    
    # Execute the C program
    try:
        print("Executing C program...")
        result = subprocess.run(command, check=True)
        print("C program executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing C program: {e}")
    time.sleep(1)
    # After the C program execution, automatically send STOP command to Arduino
    send_command_to_arduino(arduino_port, "STOP")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser for params')
    parser.add_argument('-nf', '--nframes', type=int, help='Number of frames')
    parser.add_argument('-nc', '--nchirps', type=int, help='Number of chirps in a frame, usually 182')
    parser.add_argument('-tc', '--timechirp', type=int, help='Chrip time is microseconds, usually 72')
    parser.add_argument('-s', '--samples', type=int, help='Number of ADC samples, or range bins, usually 256')
    parser.add_argument('-r', '--rate', type=int, help='Sampling rate, usually 4400')
    parser.add_argument('-tf', '--timeframe', type=int, help='Periodicity or Frame time in milliseconds')
    parser.add_argument('-p', '--pwm', type=int, help='Motor pwm value')
    parser.add_argument('-l', '--length', type=int, help='Initial length')
    parser.add_argument('-r0', '--radial', type=int, help='Initial radial distance')
    parser.add_argument('-d', '--descp', type=str, help='Data description')
    os.system("sudo macchanger --mac=c0:18:50:da:37:e0 eth0")
    os.system("sudo chmod a+rw /dev/ttyACM0")
    ans2=input("Have you connected the arduino cable to the jetson yes/no: ")
    ans1=input("Have you connected the ethernet to Jetson? yes/no: ")
    if ans1=='yes' and ans2=='yes':
        arduino_port = "/dev/ttyACM0"  # Replace with your actual port
        c_program_path = "/home/jetson/Desktop/BTP/data_collection/mmSnS/data_collect_mmwave_only"   
        now = datetime.date.today()
        date_string = str(now.strftime('%Y-%m-%d'))
        args = parser.parse_args()
        n_frames = str(args.nframes)
        n_chirps = str(args.nchirps)
        tc       = str(args.timechirp)
        adc_samples = str(args.samples)
        sampling_rate = str(args.rate)
        periodicity = str(args.timeframe)
        pwm_value = str(args.pwm)
        l = str(args.length)
        r0 = str(args.radial)
        descri = args.descp
        date_string+="_" + descri
        # n_frames=sys.argv[1]
        # n_chirps=sys.argv[2]
        # tc=sys.argv[3]
        # adc_samples=sys.argv[4]
        # sampling_rate=sys.argv[5]
        # periodicity=sys.argv[6]
        # pwm_value=sys.argv[7]
        # l=sys.argv[8]
        # r0=sys.argv[9]
        # descri=sys.argv[10]
        file_name=date_string+"_"+pwm_value+".bin"
        c_program_args=[file_name,n_frames]
        if(int(pwm_value))<=255:
            execute_c_program_and_control_arduino(arduino_port, c_program_path,c_program_args,pwm_value)
        
            bot_vel=float(input("Enter ground truth bot velocity in cm/s: "))
            ans_to_keep=input('Do you want to keep the reading? yes/no : ')
            if(ans_to_keep=='no'):
                os.system(f"rm {file_name}")
                print(f"{file_name} deleted successfully")
                sys.exit()
            os.system(f"mv {file_name} /media/jetson/93D9-AADB/")
            expected_del_phi_peak=-(3.14*bot_vel*3*86*0.00001)
            file_path="dataset.csv"
            data=[file_name,n_frames,n_chirps,tc,adc_samples,sampling_rate,periodicity,pwm_value,l,r0,descri,bot_vel,expected_del_phi_peak]
            if r0==l:
                data.append('Straight')
            else:
                data.append('Oblique')
        
            with open(file_path,'a',newline='') as file:
                writer=csv.writer(file)
                writer.writerow(data)
                print('Data appended successfully')
            
            #Push to github repo 
           # os.system("git add .");
           # os.system(f"git commit -m \"added entry for date {date_string} and {pwm_value}\"")
           # os.sytem("git push origin main")
           # print("Data pushed successfully")
        if(int(pwm_value))>255:
            send_command_to_arduino(arduino_port,"START"+pwm_value);
            time.sleep(int(int(n_frames)/5)+2)
            send_command_to_arduino(arduino_port,"STOP");

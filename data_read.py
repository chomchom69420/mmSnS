import numpy as np
import struct
import sys
import os

FRAMES = 31

dca_name = sys.argv[1]
arduino_name = sys.argv[2]
annotated_fname = sys.argv[3]
n_frames = int(sys.argv[4])

FRAMES = n_frames+1

ADC_PARAMS = {'chirps': 128,  # 32
              'rx': 4,
              'tx': 3,
              'samples': 256,
              'IQ': 2,
              'bytes': 2}

array_size = ADC_PARAMS['chirps'] * ADC_PARAMS['rx'] * ADC_PARAMS['tx'] * ADC_PARAMS['IQ'] * ADC_PARAMS['samples']
element_size = ADC_PARAMS['bytes']


def read_and_print_dca_file(filename, packet_size):
    rows = FRAMES
    cols = (728 * 1536)  # Integer division


    # Creating a numpy array of uint16 type, initialized with zeros
    frame_array = np.zeros((rows, cols), dtype=np.uint16)
    frame_time_array=np.zeros(FRAMES,dtype=np.float64)
    dirty_array=np.zeros(FRAMES)
    index=0 
    with open(filename,'rb') as file:
        last_packet_num=0
        while True:
            timestamp_data=file.read(8)
            if not timestamp_data:
                break
            timestamp=struct.unpack('d',timestamp_data)[0]
            data=file.read(packet_size)
            packet_num=struct.unpack('<1l',data[:4])[0]
            last_packet_num=packet_num
            # byte_count=struct.unpack('>Q',b'\x00\x00'+data[4:10][::-1])[0]
            if (packet_num%(1536))==0:
                print("iske baad se data read chalu karenge")
                print(packet_num)
                break
        
        packet_idx_in_frame=0
        while True:
            
            timestamp_data=file.read(8)
            if not timestamp_data:
                break
            timestamp=struct.unpack('d',timestamp_data)[0]
            
            data=file.read(packet_size) # The next packet_data
            if not data:
                break
            packet_num=struct.unpack('<1l',data[:4])[0]
            if packet_num==last_packet_num+1:
                last_packet_num=packet_num
              
                frame_array[index][packet_idx_in_frame:packet_idx_in_frame+728]= np.frombuffer(data[10:], dtype=np.uint16)
                packet_idx_in_frame+=728
                if packet_idx_in_frame==728*1535:
                # if packet_num%1536==0:
                    frame_time_array[index]=timestamp
                    
                    packet_idx_in_frame=0
                    index+=1
                continue
            elif packet_num>last_packet_num+1:
                #Packet lost ho gaya hai
                #matlab yeh frame chud gaya hai and we have to reject it
                dirty_array[index]=1
                frame_array[index][packet_idx_in_frame:packet_idx_in_frame+728]=np.zeros(728)
                packet_idx_in_frame+=728
                last_packet_num=packet_num
                if packet_idx_in_frame==728*1535:
                # if packet_num%1536==0:
                    frame_time_array[index]=timestamp
                    packet_idx_in_frame=0
                    index+=1
                continue
                
                # while (packet_num%1536)!=0:
                #     timestamp_data=file.read(8)
                #     data=file.read(packet_size)
                #     if not data:
                #         break
                #     packet_num=struct.unpack('<1l',data[:4])[0]
                # frame_time_array[index]=timestamp_data
                # index+=1
                
            # byte_count=struct.unpack('>Q',b'\x00\x00'+data[4:10][::-1])[0]

            # if c==1:
            #     print(timestamp)
            #     print(packet_num)
            
            # if (byte_count%(728*1536))==0:
            #     print("Hello")
            #     print(packet_num)
    # print(timestamp)
        for i in range(FRAMES):
            if dirty_array[i]==1:
                #reject the frame
                if i==0:
                    j=i
                    while(dirty_array[j]==0):
                        j+=1
                    frame_array[i]=frame_array[j]
                else:
                    j=i
                    while(j>=0 and dirty_array[j]==0):
                        j-=1 
                    frame_array[i]=frame_array[j]
        #Now we have a frame array proper of 100*(1456*1536))
    print("Dirtysum")
    print(np.sum(dirty_array))
    return frame_array,frame_time_array

def read_and_print_arduino_file(name):
    # Open the file for reading in binary mode
    arduino_array=np.zeros((500,3),dtype=np.float32)
    arudino_time_array=np.zeros(500,dtype=np.float64)
    index=0
    with open(arduino_name, "rb") as file:
        # Read binary data from the file
        while True:
            timestamp_data=file.read(8)
            if not timestamp_data:
                break
            left_omega_data=file.read(4)
            if not left_omega_data:
                print("Kuch load hai")
                break
            right_omega_data=file.read(4)
            if not right_omega_data:
                print("Kuch load hai")
                break

            angle_data=file.read(4)
            if not angle_data:
                print("Kuch load hai")
                break
            
            timestamp = struct.unpack('d', timestamp_data)[0]
            left_omega = struct.unpack('f', left_omega_data)[0]
            right_omega = struct.unpack('f', right_omega_data)[0]
            angle = struct.unpack('f', angle_data)[0]

            if left_omega<=0.01:
                left_omega=0.0
            if right_omega<=0.01:
                right_omega=0.0
            
            # print(right_omega,left_omega)
            arduino_array[index][0]=left_omega
            arduino_array[index][1]=right_omega
            arduino_array[index][2]=angle
            arudino_time_array[index]=timestamp
            index+=1
       
        return arduino_array,arudino_time_array
          

def annotate(dca_array,dca_time_array,ardunio_array,arduino_time_array,frames):

    if os.path.exists(annotated_fname):
        os.remove(annotated_fname)
    annotation_file = open(annotated_fname, "ab")


    i=0
    right_omega_array=np.zeros(FRAMES,dtype=np.float64)
    left_omega_array=np.zeros(FRAMES,dtype=np.float64)
    angle_array=np.zeros(FRAMES,dtype=np.float64)
    for i in range (frames):
        right_sum=0
        r_sum=0
        left_sum=0
        l_sum=0
        angle_sum=0
        cur_time=dca_time_array[i]
        for j,ele in enumerate(arduino_time_array):
            if cur_time - ele>=0.0 and cur_time-ele<1.0:
                right_sum+=arduino_array[j][1]
                left_sum+=arduino_array[j][0]
                angle_sum+=arduino_array[j][2]
                r_sum+=1
                l_sum+=1
            
        if r_sum==0:
                # pass
                # That means that frame is junk
            dca_time_array[i]=0
        else:
            right_omega_array[i]=right_sum/r_sum
            left_omega_array[i]=left_sum/l_sum
            angle_array[i] = angle_sum/r_sum

        #write to file
        annotation_file.write(dca_time_array[i])
        annotation_file.write(dca_array[i])
        annotation_file.write(left_omega_array[i])
        annotation_file.write(right_omega_array[i])
        annotation_file.write(angle_array[i])
        size_dca_time_array = dca_time_array[i].nbytes
        size_dca_array = dca_array[i].nbytes
        size_left_omega_array = left_omega_array[i].nbytes
        size_right_omega_array = right_omega_array[i].nbytes
        size_angle_array = angle_array[i].nbytes

        #print(size_dca_time_array)
        #print(size_dca_array)
        #print(size_left_omega_array)
        #print(size_left_omega_array)
        #print(size_angle_array)

 

    
    annotation_file.close()

    return dca_array, dca_time_array, right_omega_array, left_omega_array, angle_array

dca_array,dca_time_array=read_and_print_dca_file(dca_name,1466)
arduino_array,arduino_time_array=read_and_print_arduino_file(arduino_name)
dca_array,dca_time_array,right_omega,left_omega,angle_array=annotate(dca_array,dca_time_array,arduino_array,arduino_time_array,FRAMES)
 
for i in range(FRAMES):
    # print(dca_time_array[i], left_omega[i], right_omega[i])
    print(dca_time_array[i], arduino_time_array[i],right_omega[i],left_omega[i],angle_array[i])

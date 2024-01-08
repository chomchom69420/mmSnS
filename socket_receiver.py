import struct
import numpy as np
import socket
import sys
import pickle
import time
ADC_PARAMS = {
    'chirps': 128,
    'rx': 4,
    'tx': 3,
    'samples': 256,
    'IQ': 2,
    'bytes': 2
}

BYTES_IN_PACKET = 1456
BYTES_IN_FRAME = (
    ADC_PARAMS['chirps'] * ADC_PARAMS['rx'] * ADC_PARAMS['tx'] *
    ADC_PARAMS['IQ'] * ADC_PARAMS['samples'] * ADC_PARAMS['bytes']
)
BYTES_IN_FRAME_CLIPPED = (BYTES_IN_FRAME // BYTES_IN_PACKET) * BYTES_IN_PACKET
PACKETS_IN_FRAME = BYTES_IN_FRAME / BYTES_IN_PACKET
PACKETS_IN_FRAME_CLIPPED = BYTES_IN_FRAME // BYTES_IN_PACKET
UINT16_IN_PACKET = BYTES_IN_PACKET // 2
UINT16_IN_FRAME = BYTES_IN_FRAME // 2

static_ip = '192.168.33.30'
data_port = 4098
data_recv = (static_ip, data_port)

# open a file
name = sys.argv[1]
num_frames = int(sys.argv[2])

n_packets = (num_frames + 1) * 1536

data_socket = socket.socket(
    socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
all_packet_data = bytearray()
# Bind data socket to fpga
data_socket.bind(data_recv)
c = 0
last_packet_num = 0
lost_count = 0
f = open(name + ".bin", 'ab')
start_time=time.time()
# List of dictionaries to store packet_num and packet_data
for c in range(n_packets):
    data, addr = data_socket.recvfrom(4000)
    t=time.time()-start_time
    packet_num = struct.unpack('<1l', data[:4])[0]
    # byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
    
    # Add the structured data to the buffer
    # all_packet_data.extend(struct.pack('<1lQ', packet_num, byte_count))
    all_packet_data.extend(struct.pack('d',t))
    all_packet_data.extend(data)
    if c == 0:
        last_packet_num = packet_num
    elif last_packet_num < packet_num - 1:
        lost_count += 1
    last_packet_num = packet_num

    # Append dictionary to the list

# Write the list of dictionaries to the file
print(lost_count)
# Write all data to the file at once
with open(name + ".bin", "wb") as file:
    file.write(all_packet_data)


def read_and_print_file(filename, packet_size):
    c=0
    with open(filename,'rb') as file:
        while True:
            timestamp_data=file.read(8)
            if not timestamp_data:
                break
            timestamp=struct.unpack('d',timestamp_data)[0]
            data=file.read(packet_size)
            packet_num=struct.unpack('<1l',data[:4])[0]
            byte_count=struct.unpack('>Q',b'\x00\x00'+data[4:10][::-1])[0]
            if (byte_count%(1456*1536))==0:
                print("Hello")
                print(packet_num)
                break

        while True:
            timestamp_data=file.read(8)
            if not timestamp_data:
                break
            timestamp=struct.unpack('d',timestamp_data)[0]
            c+=1
            data=file.read(packet_size)
            if not data:
                break
            packet_num=struct.unpack('<1l',data[:4])[0]
            byte_count=struct.unpack('>Q',b'\x00\x00'+data[4:10][::-1])[0]
            if c==1:
                print(timestamp)
                print(packet_num)
            
            if (byte_count%(1456*1536))==0:
                print("Hello")
                print(packet_num)
    print(timestamp)
# read_and_print_file(name+'.bin',1466)

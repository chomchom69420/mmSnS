import struct

with open('test.bin', 'rb') as file:
    for _ in range(100):
        time_info = struct.unpack('q', file.read(8))[0]
        print("Time:", time_info)
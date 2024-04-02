import struct

# Open the binary file for reading in binary mode
with open('test.bin', 'rb') as file:
    # Read the binary data in chunks of 8 bytes (since each entry is 8 bytes long)
    for _ in range(100):
        # Read 8 bytes and unpack them as a single long integer (assuming the time information is stored as a long integer)
        time_info = struct.unpack('q', file.read(8))[0]
        
        # Print the time information
        print("Time:", time_info)
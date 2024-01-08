#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/socket.h>
#include <time.h>
#include <stdbool.h>
#include <sys/time.h>


#define CHIRPS 128
#define RX 4
#define TX 3
#define SAMPLES 256
#define IQ 2
#define BYTES 2
#define BYTES_IN_PACKET 1456
#define BYTES_IN_FRAME (CHIRPS * RX * TX * IQ * SAMPLES * BYTES)
#define BYTES_IN_FRAME_CLIPPED ((BYTES_IN_FRAME / BYTES_IN_PACKET) * BYTES_IN_PACKET)
#define PACKETS_IN_FRAME (BYTES_IN_FRAME / BYTES_IN_PACKET)
#define PACKETS_IN_FRAME_CLIPPED (BYTES_IN_FRAME / BYTES_IN_PACKET)
#define UINT16_IN_PACKET (BYTES_IN_PACKET / 2)
#define UINT16_IN_FRAME (BYTES_IN_FRAME / 2)

#define STATIC_IP "192.168.33.30"
#define DATA_PORT 4098

// void read_and_print_file(const char *filename, size_t packet_size) {
//     FILE *read_file = fopen(filename, "rb");
    
//     if (read_file == NULL) {
//         perror("fopen");
//         return;
//     }

//     if (remove("data.bin") == 0) {
//         printf("File '%s' deleted successfully.\n", "data.bin");
//     } 
//     else {
//         perror("Error deleting file");
//     }

//     FILE *write_file = fopen("data.bin", "ab");
//     if (write_file == NULL) {
       
//         perror("fopen");
//         return;
//     }

//     printf("Here");

//     //Declare last_frame_array
//     uint8_t last_frame_array[1536][1456];

//     printf("Here1");

//     //Declare current_frame_array
//     uint8_t current_frame_array[1536][1456];
//      printf("Here2");

//     //Declare packet in frame index
//     uint64_t packet_in_frame_idx = 0;

//     //Declare a skip flag for packet loss
//     bool skip_flag = false;

//     int c = 0;
//     int frame_num = -1;
//     uint64_t first_frame_pckt_num=0;
//     uint64_t last_packet_num=0;
//    //will be goiod if we initialise the tme stamp outsid ethe loop
//     while (1) {
//         double timestamp;
//         size_t bytes_read = fread(&timestamp, sizeof(double), 1, read_file);
//         //This is the time stamp
//         if (bytes_read != 1) {
//             break;
//         }

//         char data[packet_size];
//         bytes_read = fread(data, 1, packet_size, read_file);
//         if (bytes_read != packet_size) {
//             break;
//         }

//         //What i feel is that the above code if only for checking is there is data p[resent or not]
//         //isnt data array getting the same values every iteration fread(data, 1, packet_size, read_file) NO
//         uint64_t packet_num;
//         memcpy(&packet_num, data, 4);
//         uint64_t byte_count;
//         memcpy(&byte_count, data + 4, 6);
//         byte_count = be64toh(byte_count);

//         //This part of the code replaces lost frames by previous frames (add timestamping in write_file)

//         //Check if packet_num % 1536==0
//         //If yes, store the current frame array in the last frame array and clean the
//         //current frame index
//         //Store the current frame array into the write file

//         //If no, check if packet lost
//             //If yes, copy the last frame array into the current frame array and skip all the rest of the packets till next %1536 ==0 (skip_flag)
//             //If no, keep appending into the current frame array

//         if(c==0) {
//             last_packet_num = packet_num; 
//             c++;
//         }

//         if (packet_num % 1536 == 0) {
            
//             if(frame_num==-1) {
                
//                 first_frame_pckt_num = packet_num;
//                 frame_num = 0;

//             }
//             else 
//             {
//                 fwrite(&timestamp, sizeof(double), 1, write_file);
//                 fwrite(current_frame_array, 1, 1456*1536, write_file);
//                 memcpy(last_frame_array, current_frame_array, 1456*1536);
//                 packet_in_frame_idx = 0;
//             }
//             // printf("%ld\n", packet_num);
            
//             //Clear the skip flag
//             skip_flag = false;
//         }
//         else 
//         {
//             //Check if skip flag is false
//             if(!skip_flag)
//             {
//                 if(last_packet_num < packet_num -1) 
//                 {
//                     //Packet lost, copy last frame to current frame
//                     memcpy(current_frame_array, last_frame_array, 1456*1536);

//                     //Set skip flag
//                     skip_flag = true;
//                 }
//                 else 
//                 {
//                     //Append into the current frame array
//                     memcpy((current_frame_array + packet_in_frame_idx*1456), data + 10, 1456);
//                     printf("%ld", packet_in_frame_idx);
//                     packet_in_frame_idx+=1;
//                 }
//             }

//         }

//         last_packet_num = packet_num;
//     }

//     fclose(read_file);
//     fclose(write_file);
// }

int main(int argc, char *argv[]) {
    
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <filename> <num_frames>\n", argv[0]);
        return 1;
    }
    
    char *name = argv[1];
    int num_frames = atoi(argv[2]);
    int n_packets = (num_frames + 1) * 1536;

    int data_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (data_socket == -1) {
        perror("socket");
        return 1;
    }
    
    struct sockaddr_in data_recv;
    data_recv.sin_family = AF_INET;
    data_recv.sin_port = htons(DATA_PORT);
    inet_aton(STATIC_IP, &data_recv.sin_addr);

    if (bind(data_socket, (struct sockaddr *)&data_recv, sizeof(data_recv)) == -1) {
        perror("bind");
        close(data_socket);
        return 1;
    }


    

    strcat(name, ".bin");
    
    if(access(name, F_OK) != -1) //File exists
    {
        if (remove(name) == 0) {
            printf("file %s is removed\n",name);
        } 
        else {
            perror("Error deleting file\n");
        }
    }
  
    int c = 0;
    int last_packet_num = 0;
    int lost_count = 0;
    FILE *file = fopen(name, "ab");
    if (file == NULL) {
        
        perror("fopen");
        close(data_socket);
        return 1;
    }
     
    struct timeval start_time;
    gettimeofday(&start_time, NULL);
    printf("Hello");
    // // List of dictionaries to store packet_num and packet_data
    for (c = 0; c < n_packets; c++) {
        char data[4000];
        struct sockaddr_in addr;
        socklen_t addr_len = sizeof(addr);
        ssize_t bytes_received = recvfrom(data_socket, data, sizeof(data), 0, (struct sockaddr *)&addr, &addr_len);
        // ssize_t bytes_received =1466;
        
        if (bytes_received == -1) {
            perror("recvfrom");
            break;
        }

        struct timeval current_time;
        gettimeofday(&current_time, NULL);
        // double t = (current_time.tv_sec - start_time.tv_sec) + (current_time.tv_usec - start_time.tv_usec) / 1e6;
        long long t = time(NULL);

        uint64_t packet_num;
        memcpy(&packet_num, data, 4);
        // printf("%ld\n",packet_num);

        fwrite(&t, sizeof(long long), 1, file);
        fwrite(data, 1, bytes_received, file);

        if (c == 0) {
            last_packet_num = packet_num;
        } else if (last_packet_num < packet_num - 1) {
            lost_count += 1;
        }
        last_packet_num = packet_num;
    }

    // Write the list of dictionaries to the file
    printf("%d\n", lost_count);
    fclose(file);
    close(data_socket);

    // return 0;
}



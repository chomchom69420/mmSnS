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
#include "socket_receiver.h"

void get_sensor_data(const char* filename, int num_frames) {
    int n_packets = (num_frames + 1) * 1536;

    int data_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (data_socket == -1) {
        perror("socket");
        return;
    }
    
    struct sockaddr_in data_recv;
    data_recv.sin_family = AF_INET;
    data_recv.sin_port = htons(DATA_PORT);
    inet_aton(STATIC_IP, &data_recv.sin_addr);

    if (bind(data_socket, (struct sockaddr *)&data_recv, sizeof(data_recv)) == -1) {
        perror("bind");
        close(data_socket);
        return;
    }
    
    if(access(filename, F_OK) != -1) //File exists
    {
        if (remove(filename) == 0) {
            printf("file %s is removed\n",filename);
        } 
        else {
            perror("Error deleting file\n");
        }
    }
  
    int c = 0;
    int last_packet_num = 0;
    int lost_count = 0;
    FILE *file = fopen(filename, "ab");
    if (file == NULL) {
        
        perror("fopen");
        close(data_socket);
        return;
    }
     
    struct timeval start_time;
    gettimeofday(&start_time, NULL);

    //Loop over all packets and receive
    for (c = 0; c < n_packets; c++) {
        
        char data[4000];
        struct sockaddr_in addr;
        socklen_t addr_len = sizeof(addr);

        //Receiving the data from the data socket. If successful, 1466 bytes of data should be received in the packet
        //stored in the variable data
        ssize_t bytes_received = recvfrom(data_socket, data, sizeof(data), 0, (struct sockaddr *)&addr, &addr_len);
        
        if (bytes_received == -1) {
            perror("recvfrom");
            break;
        }

        //Store current time in epochs for timestamping
        time_t curr_time;
        time(&curr_time);

        uint64_t packet_num;
        memcpy(&packet_num, data, 4);

        //Store time and data of a packet in the bin file
        fwrite(&curr_time, sizeof(time_t), 1, file);
        fwrite(data, 1, bytes_received, file);

        
        if (c == 0) {
            last_packet_num = packet_num;
        }
        //If packet is lost, then packet number will be skipped, increment lost_count
        //Nothing is done with lost packets here, just to print. Lost packets are dealt with in data_read.py
        else if (last_packet_num < packet_num - 1) {
            lost_count += 1;
        }
        last_packet_num = packet_num;
    }

    //Print out the number of lost packets in the entire data
    printf("%d\n", lost_count);
    fclose(file);
    close(data_socket);
}

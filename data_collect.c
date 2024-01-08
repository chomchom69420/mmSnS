#include "socket_receiver.h"
#include "arduino_receiver.h"

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

// Structure to hold parameters for threadFunction1
struct ThreadParams1 {
    char *arduino_filename;
    char *port_name;
    time_t duration;
};

// Structure to hold parameters for threadFunction2
struct ThreadParams2 {
    char *sensor_filename;
    int numframes;
};


void* threadFunction1(void* arg) {

    struct ThreadParams1* params = (struct ThreadParams1*)arg;
    char *arduino_filename = params->arduino_filename;
    char *port_name = params->port_name;
    time_t duration = params->duration;
    
    get_arduino_data(arduino_filename, port_name, duration);

    pthread_exit(NULL);
}


void* threadFunction2(void* arg) {
    
    struct ThreadParams2* params = (struct ThreadParams2*)arg;
    const char *sensor_filename = params->sensor_filename;
    int numframes = params->numframes;

    get_sensor_data(sensor_filename, numframes);

    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {

    if(argc!=6) {
        fprintf(stderr, "Usage: %s <sensor_filename>.bin <num_frames> <arduino_filename>.bin <portname> <duration>(s)\n", argv[0]);
        return 1;
    }

    //Sensor inputs
    char *sensor_filename = argv[1];
    int numframes = atoi(argv[2]);

    //Arduino inputs
    char *arduino_filename = argv[3];
    char *port_name = argv[4];
    time_t duration = atoi(argv[5]);

    struct ThreadParams1 params1 = {arduino_filename, 
                                    port_name,
                                    duration};

    struct ThreadParams2 params2 = {sensor_filename, 
                                    numframes};

    pthread_t tid1, tid2; 

    // Creating the first thread
    if (pthread_create(&tid1, NULL, threadFunction1, (void*)&params1) != 0) {
        fprintf(stderr, "Error creating thread 1\n");
        return 1;
    }

    // Creating the second thread
    if (pthread_create(&tid2, NULL, threadFunction2, (void*)&params2) != 0) {
        fprintf(stderr, "Error creating thread 2\n");
        return 1;
    }

    // Wait for the threads to finish
    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);

    printf("Both threads have finished\n");

    return 0;
}

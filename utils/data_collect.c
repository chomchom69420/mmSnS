#include "socket_receiver.h"
#include "arduino_receiver.h"

#include <stdlib.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

// #define ARDUINO 1

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

#ifdef ARDUINO
#ifdef _WIN32
DWORD WINAPI threadFunction1(LPVOID arg) {
#else
void* threadFunction1(void* arg) {
#endif
    struct ThreadParams1* params = (struct ThreadParams1*)arg;
    char *arduino_filename = params->arduino_filename;
    char *port_name = params->port_name;
    time_t duration = params->duration;

    get_arduino_data(arduino_filename, port_name, duration);

#ifdef _WIN32
    return 0;
#else
    pthread_exit(NULL);
#endif
}
#endif


#ifdef _WIN32
DWORD WINAPI threadFunction2(LPVOID arg) {
#else
void* threadFunction2(void* arg) {
#endif
    struct ThreadParams2* params = (struct ThreadParams2*)arg;
    const char *sensor_filename = params->sensor_filename;
    int numframes = params->numframes;

    get_sensor_data(sensor_filename, numframes);

#ifdef _WIN32
    return 0;
#else
    pthread_exit(NULL);
#endif
}

int main(int argc, char *argv[]) {

    #ifdef ARDUINO
    if(argc!=6) {
        fprintf(stderr, "Usage: %s <sensor_filename>.bin <num_frames> <arduino_filename>.bin <portname> <duration>(s)\n", argv[0]);
        return 1;
    }
    #else
    if(argc!=3) {
        fprintf(stderr, "Usage: %s <sensor_filename>.bin <num_frames>\n", argv[0]);
        return 1;
    }
    #endif

    //Sensor inputs
    char *sensor_filename = argv[1];
    int numframes = atoi(argv[2]);

    #ifdef ARDUINO
    //Arduino inputs
    char *arduino_filename = argv[3];
    char *port_name = argv[4];
    time_t duration = atoi(argv[5]);

    struct ThreadParams1 params1 = {arduino_filename, 
                                    port_name,
                                    duration};
    #endif

    struct ThreadParams2 params2 = {sensor_filename, 
                                    numframes};

    #ifdef ARDUINO
    #ifdef _WIN32
    HANDLE hThread1;
    #else
    pthread_t tid1;
    #endif
    #endif

    #ifdef _WIN32
    HANDLE hThread2;
    #else
    pthread_t tid2; 
    #endif

    #ifdef ARDUINO
    // Creating the first thread
    #ifdef _WIN32
    hThread1 = CreateThread(NULL, 0, threadFunction1, &params1, 0, NULL);
    if (hThread1 == NULL) {
        fprintf(stderr, "Error creating thread 1\n");
        return 1;
    }
    #else
    if (pthread_create(&tid1, NULL, threadFunction1, (void*)&params1) != 0) {
        fprintf(stderr, "Error creating thread 1\n");
        return 1;
    }
    #endif
    #endif

    // Creating the second thread
    #ifdef _WIN32
    hThread2 = CreateThread(NULL, 0, threadFunction2, &params2, 0, NULL);
    if (hThread2 == NULL) {
        fprintf(stderr, "Error creating thread 2\n");
        return 1;
    }
    #else
    if (pthread_create(&tid2, NULL, threadFunction2, (void*)&params2) != 0) {
        fprintf(stderr, "Error creating thread 2\n");
        return 1;
    }
    #endif

    // Wait for the threads to finish
    #ifdef ARDUINO
    #ifdef _WIN32
    WaitForSingleObject(hThread1, INFINITE);
    CloseHandle(hThread1);
    #else
    pthread_join(tid1, NULL);
    #endif
    #endif

    #ifdef _WIN32
    WaitForSingleObject(hThread2, INFINITE);
    CloseHandle(hThread2);
    #else
    pthread_join(tid2, NULL);
    #endif

    #ifdef ARDUINO
    printf("Both threads have finished\n");
    #else
    printf("Sensor thread has completed\n");
    #endif

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
    #include <windows.h>
    #include <io.h>  // For _access and _unlink in Windows
    #define ACCESS _access
    #define REMOVE _unlink
    #define F_OK 0  // F_OK equivalent in Windows
#else
    #include <fcntl.h>
    #include <termios.h>
    #include <sys/time.h>
    #include <unistd.h>  // For access and unlink in Linux
    #define ACCESS access
    #define REMOVE remove
#endif

// #include <time.h>
#include "arduino_receiver.h"

#ifdef _WIN32
// Windows-specific function to replace gettimeofday()
int gettimeofday(struct timeval *tp, struct timezone *tzp)
{
    FILETIME fileTime;
    ULARGE_INTEGER ull;
    uint64_t t;
    GetSystemTimeAsFileTime(&fileTime);
    ull.LowPart = fileTime.dwLowDateTime;
    ull.HighPart = fileTime.dwHighDateTime;
    t = ull.QuadPart / 10ULL - 11644473600000000ULL;
    tp->tv_sec = (long)(t / 1000000ULL);
    tp->tv_usec = (long)(t % 1000000ULL);
    return 0;
}
#endif

void get_arduino_data(const char* filename, const char* portname, time_t duration) {

#ifdef _WIN32
    // Open serial port
    HANDLE serial_port = CreateFileA(portname, GENERIC_READ | GENERIC_WRITE, 0, 0, OPEN_EXISTING, 0, 0);
    if (serial_port == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Error opening serial port\n");
        return;
    }

    // Configure serial port
    DCB dcbSerialParams = {0};
    dcbSerialParams.DCBlength = sizeof(dcbSerialParams);
    if (!GetCommState(serial_port, &dcbSerialParams)) {
        fprintf(stderr, "Error getting serial port state\n");
        CloseHandle(serial_port);
        return;
    }

    dcbSerialParams.BaudRate = CBR_115200;  // Set baud rate
    dcbSerialParams.ByteSize = 8;
    dcbSerialParams.StopBits = ONESTOPBIT;
    dcbSerialParams.Parity = NOPARITY;

    if (!SetCommState(serial_port, &dcbSerialParams)) {
        fprintf(stderr, "Error setting serial port state\n");
        CloseHandle(serial_port);
        return;
    }

    COMMTIMEOUTS timeouts = {0};
    timeouts.ReadIntervalTimeout = 50;  // 50ms interval timeout
    timeouts.ReadTotalTimeoutConstant = 50;
    timeouts.ReadTotalTimeoutMultiplier = 10;

    if (!SetCommTimeouts(serial_port, &timeouts)) {
        fprintf(stderr, "Error setting serial port timeouts\n");
        CloseHandle(serial_port);
        return;
    }

#else
    int serial_port = open(portname, O_RDWR);
#endif

    if (ACCESS(filename, F_OK) != -1) {  // File exists
        if (REMOVE(filename) == 0) {
            printf("file %s is removed\n", filename);
        } else {
            perror("Error deleting file\n");
        }
    }

    FILE *file = fopen(filename, "ab");
    if (file == NULL) {
        perror("fopen");

#ifdef _WIN32
        CloseHandle(serial_port);
#else
        close(serial_port);
#endif
        return;
    }

    struct timeval start_time;
    gettimeofday(&start_time, NULL);

#ifdef _WIN32
    if (serial_port == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Error opening serial port\n");
        return;
    }
#else
    if (serial_port < 0) {
        perror("Error opening serial port");
        return;
    }

    struct termios tty;
    if (tcgetattr(serial_port, &tty) != 0) {
        perror("Error getting serial port attributes");
        close(serial_port);
        return;
    }

    cfsetospeed(&tty, B115200);  // Adjust the baud rate as needed
    cfsetispeed(&tty, B115200);

    tty.c_cflag |= (CLOCAL | CREAD);  // Enable receiver and ignore modem control lines

    // Set the character size
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;  // 8-bit characters

    // No parity
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;  // 1 stop bit

    // Raw input
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);

    // Read at least one character or block for 100 ms
    tty.c_cc[VMIN] = 1;
    tty.c_cc[VTIME] = 1;

    if (tcsetattr(serial_port, TCSANOW, &tty) != 0) {
        perror("Error setting serial port attributes");
        close(serial_port);
        return;
    }
#endif

    // Read data from the serial port
    char buffer[1024];
    DWORD bytesRead;
    
    time_t startTime = time(NULL);

    int count = 0;

    while (time(NULL) - startTime < duration) {
#ifdef _WIN32
        if (!ReadFile(serial_port, buffer, sizeof(buffer), &bytesRead, NULL)) {
            fprintf(stderr, "Error reading from serial port\n");
            break;
        }
#else
        bytesRead = read(serial_port, buffer, sizeof(buffer));
#endif
        if (bytesRead > 0) {

            count++;

            // Storing current time as epochs for timestamping
            long long t = time(NULL);

            float left_omega, right_omega, angle;

            // Parse and store variables from the UART buffer
            sscanf(buffer, "L: %f R: %f A:%f\n", &left_omega, &right_omega, &angle);

            // Write values to the file with timestamp
            fwrite(&t, sizeof(long long), 1, file);
            fwrite(&right_omega, sizeof(double), 1, file);
            fwrite(&left_omega, sizeof(double), 1, file);
            fwrite(&angle, sizeof(double), 1, file);
        }
#ifdef _WIN32
        else if (bytesRead == 0) {
            fprintf(stderr, "Error reading from serial port\n");
            break;
        }
#else
        else if (bytesRead < 0) {
            perror("Error reading from serial port");
            break;
        }
#endif
    }

    // Close the serial port
#ifdef _WIN32
    CloseHandle(serial_port);
#else
    close(serial_port);
#endif
    fclose(file);

    // Print number of datapoints read
    printf("Count = %d\n", count);
}
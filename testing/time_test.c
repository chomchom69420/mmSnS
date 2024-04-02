#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    FILE *file = fopen("test.bin", "ab");
    if (file == NULL) {
        perror("fopen");
        return 0;
    }

    for(int i=0;i<100;i++) {
        time_t curr_time;
        time(&curr_time);
        fwrite(&curr_time, sizeof(time_t), 1, file);
    }

    printf("Program has executed.");

    return 0;
}
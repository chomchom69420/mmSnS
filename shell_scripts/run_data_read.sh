#!/bin/bash

# Define the Python file and arguments
python_file="data_read.py"

# Define the number of times to execute the Python file
num_executions=50

# Loop through and execute the Python file with specified arguments
for ((i=1; i<=$num_executions; i++)); do
    echo "Executing iteration $i"
    
    dca_filename="test$i.bin"
    arduino_filename="arduino$i.bin"
    data_filename="data$i.bin"
    num_frames="60"

    python3 "$python_file" "$dca_filename" "$arduino_filename" "$data_filename" "$num_frames"
done

echo "Script completed."

#!/bin/bash

# Define the source directory containing files to be copied
source_directory="./"

# Define the destination directory on the remote host
destination_directory="/home/soham/Desktop/BTP_mmwave"

# Define the remote host and username
remote_host="soham@192.168.0.4"
password="khinda113"

num_executions=50

for ((i=1; i<=$num_executions; i++)); do
    echo "Sending file data$i.bin"

    file="data$i.bin"

    sshpass -p "$password" scp "$source_directory/$file" "$remote_host:$destination_directory"
done

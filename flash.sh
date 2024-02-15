#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <COM port>"
    exit 1
fi

# Argument to be passed to the flash command
COM_PORT="$1"

#PlatformIO directory path
DIR_PATH="/home/sohamc1909/Documents/PlatformIO/Projects/sns_bot"

# Path to the virtual environment
VENV_PATH="/home/sohamc1909/.platformio/penv"

# Check if the virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

echo "Virtual environment activated."

pio run -e uno -t upload -d "$DIR_PATH" --upload-port "$COM_PORT"

echo "Code flashed."


#!/bin/bash

# Kill any existing Xvfb
killall Xvfb || true

# Start Xvfb with NVIDIA device
Xvfb :99 -screen 0 1024x768x24 +extension GLX +render -noreset &
export DISPLAY=:99

# Wait for Xvfb to start
sleep 2

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate roboexp

# Run the Python script
python interactive_explore.py

# Clean up
killall Xvfb 
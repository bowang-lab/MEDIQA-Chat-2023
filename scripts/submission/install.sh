#!/bin/bash

# Create and activate the virtual environment
VENV="wanglab_venv"
python3 -m venv "$VENV"
source "$VENV"

# Clone the repo and install any dependencies
pip install -r requirements.txt

# Finally, deactivate the virtual environment
deactivate
#!/bin/bash

# Parse command-line arguments
while getopts "d:m:" opt; do
  case $opt in
    d) CUDA_VISIBLE_DEVICES=$OPTARG ;;   # Device number
    m) MODEL=$OPTARG ;;                   # Model name (e.g., "resnet")
    *) echo "Usage: $0 [-d CUDA_DEVICE] [-m MODEL_NAME]" >&2
       exit 1 ;;
  esac
done

# Default values
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # Default to GPU 0 if not provided
MODEL=${MODEL:-resnet}  # Default to "resnet" if not provided

# Train the model for the specified configuration
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m src.train -m tags="[$MODEL,lbl]" model=lbl_$MODEL data.fold=0,1,2,3,4
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m src.train -m tags="[$MODEL,bra]" model=lbl_$MODEL data.fold=0,1,2,3,4 data.dataset_name=BraTs_TCGA_2d trainer.max_epochs=10

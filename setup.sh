#!/bin/bash

module load anaconda/3
module load cuda/11.8

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source "$(conda info --base)/etc/profile.d/conda.sh"
# Create if not already exist: conda create -n faithfulness python=3.10
conda activate myenv

export BNB_CUDA_VERSION=118

pip install torch==2.2.0+cu118 \
            torchvision==0.17.0+cu118 \
            torchaudio==2.2.0+cu118 \
            --extra-index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

python -c "import torch; import bitsandbytes as bnb; print(f'Torch: {torch.__version__} | CUDA: {torch.version.cuda} | GPU Available: {torch.cuda.is_available()}')"
@echo off

set CUDA_VISIBLE_DEVICES=0

python download_ffhq_aging.py --resolution 256
python run_deeplab.py --resolution 256

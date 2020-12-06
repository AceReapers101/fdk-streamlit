#!/usr/bin/env bash

virtualenv venv
source ./venv/bin/activate
git clone https://github.com/1stDayHack/FDK.git
cp -r ./FDK/* ./
rm -rf ./FDK/
pip install -r requirements.txt
pip install streamlit
apt-get update
apt-get install -y libgl1-mesa-dev
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.7/index.html
pip install gdown
gdown https://drive.google.com/u/0/uc?id=1lq5UcxRWnZ5XUQzYD2lW2vFYBBFHdLff
cp RRDB_ESRGAN_x4.pth src/core/base_libs/ESRGAN/models/
rm RRDB_ESRGAN_x4.pth
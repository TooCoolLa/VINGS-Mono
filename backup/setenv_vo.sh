conda create --name vings_vo python=3.9.19
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter==2.0.2 -f https://data.pyg.org/whl/torch-2.0.2+cu118.html
pip install -r requirements.txt

cmake ./submodules/gtsam -DGTSAM_BUILD_PYTHON=1 -B build_gtsam 
cmake --build build_gtsam --config RelWithDebInfo -j
cd build_gtsam
make python-install

cd ../submodules/dbef

cd submodules/dbef/thirdparty
git clone --recursive https://github.com/princeton-vl/lietorch
git clone --recursive https://gitlab.com/libeigen/eigen.git
python setup.py install
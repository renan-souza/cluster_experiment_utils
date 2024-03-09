# Got from here: https://gist.github.com/briansp2020/717f5cab006889f056b36eacee7dd5d7
#
# Build PyTorch
git clone https://github.com/ROCmSoftwarePlatform/pytorch.git
cd pytorch
git submodule init
git submodule update
python tools/amd_build/build_pytorch_amd.py
python tools/amd_build/build_caffe2_amd.py
USE_ROCM=1 MAX_JOBS=16 python setup.py install
pip install matplotlib torchvision torchtext
pip install datasets
pip install experiment_cluster_utils
pip install shap # conflicts with requirements-responsible_ai.txt @ flowcept
pip install flowcept[dask,analytics,amd]

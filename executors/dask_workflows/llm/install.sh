pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2
pip install experiment_cluster_utils
pip install shap # conflicts with requirements-responsible_ai.txt @ flowcept
pip install flowcept[dask,analytics,amd] # or in flowcept's directory,  pip install -e .[dask,analytics,and]
pip list | grep torch
python -c 'import torch; print(torch.cuda.is_available())'

```shell
$ (base) conda create -n flowcept_llm_tests python=3.8 -y
$ (base) conda activate flowcept_llm_tests
$ (flowcept_llm_tests) pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2   # [1,2,3]
```
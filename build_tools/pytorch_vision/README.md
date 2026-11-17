# PyTorch Vision rocAL image

Build the rocAL 2.5.0 image matched to ROCm 7.2.4:

```bash
docker build \
  -f build_tools/pytorch_vision/Dockerfile.rocal-7.2.4 \
  -t cvs/pytorch-rocal:rocm7.2.4-py3.12-torch2.10-rocal2.5.0 \
  .
```

The Dockerfile pins:

- ROCm 7.2.4
- Python 3.12
- PyTorch 2.10
- torchvision 0.25
- rocAL 2.5.0

The AMD graphics repository is required because rocAL's rocDecode/rocJPEG
dependencies require `mesa-amdgpu-va-drivers`, which is not in the base
PyTorch image's ROCm-only apt source.

Verify a GPU ImageNet batch with the script and command documented in
`cvs/input/config_file/training/pytorch_vision/README.md`.

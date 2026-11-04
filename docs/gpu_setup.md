# GPU setup: RTX 5090 on Windows 10

Training, silver-data generation and evaluation run on one NVIDIA GeForce RTX 5090
(32 GB, Blackwell architecture, compute capability 12.0, also written `sm_120`) in a
Windows 10 PC. The Python code runs in Linux, either in WSL2 Ubuntu or in a Docker
container. This guide covers both.

Blackwell needs software built for it. A PyTorch build for CUDA 12.6 or older has no
`sm_120` kernels and fails with `no kernel image is available for execution on the
device`. Builds for CUDA 12.8 or newer include them.

## Versions

Checked together in November 2026:

| Component | Version | Why this version |
| --- | --- | --- |
| NVIDIA driver (Windows) | R580 or newer | CUDA 13 programs need an R580+ driver. |
| PyTorch | 2.14.1, CUDA 13.0 build (`cu130`) | Includes `sm_120` kernels. Newest release that Unsloth 2026.11.2 accepts (it requires `torch<2.15`). |
| Unsloth, unsloth_zoo | 2026.11.2 | Supports RTX 50-series GPUs. |
| Transformers | 5.17.0 | Newest release that Unsloth 2026.11.2 accepts. |
| TRL | 1.15.0 | Newest release that Unsloth 2026.11.2 accepts. |
| PEFT | 0.21.2 | Latest release. |
| bitsandbytes | 0.50.2 | Ships a CUDA 13.0 binary; used only by the 4-bit QLoRA variant. |
| Training image base | `pytorch/pytorch:2.14.1-cuda13.0-cudnn9-devel` | Same PyTorch build as the local setup. |
| vLLM server image | `vllm/vllm-openai:v0.33.0` | CUDA 13.0.2, built for compute capability 7.5 to 12.0. |

The training extras in `pyproject.toml` (`.[train]`) pin these library versions.
PyTorch is installed separately because the CUDA build is chosen at install time.

The PyTorch index for CUDA 12.8 (`whl/cu128`) stops at PyTorch 2.11. PyTorch 2.14.1 is
available for CUDA 12.6 (no Blackwell support), 13.0 and 13.2. CUDA 13.0 is used here
because it matches the vLLM image.

## 1. Windows host

1. Use Windows 10 version 21H2 or later (run `winver` to check). GPU access from WSL2
   needs it. Version 22H2 is the last Windows 10 release.
2. Install the latest NVIDIA Game Ready or Studio driver for the RTX 5090 from
   [nvidia.com/drivers](https://www.nvidia.com/drivers). It must be R580 or newer.
   NVIDIA's final Game Ready driver with feature updates for Windows 10 came out in
   October 2026. After that, Windows 10 gets quarterly security driver updates until
   October 2029. The October 2026 drivers (version 617 and later) are well above R580.
3. Install only the Windows driver. WSL2 and Docker use it through a passthrough
   library. Do not install an NVIDIA Linux driver inside WSL2, because it would replace
   that library.
4. Install WSL2 and Ubuntu 24.04 from an administrator PowerShell, then restart if asked:

   ```powershell
   wsl --install -d Ubuntu-24.04
   wsl --update
   wsl --set-default-version 2
   ```

5. Give WSL2 enough memory. By default it gets half of the host RAM. A 4B model is
   about 8 GB in bf16, and loading or merging it can briefly need twice that in CPU
   memory. Create `%UserProfile%\.wslconfig` (sizes for a 64 GB host; adjust to yours):

   ```ini
   [wsl2]
   memory=48GB
   swap=16GB
   ```

   Then run `wsl --shutdown` and open Ubuntu again.

## 2. WSL2 Ubuntu

Check that the GPU is visible:

```bash
nvidia-smi
```

The table should list the RTX 5090. `CUDA Version: 13.x` in the header is the newest
CUDA version the driver supports.

Keep the repository in the Linux file system (for example `~/explain-to-learn`), not
under `/mnt/c`. Model files and datasets load much more slowly through `/mnt/c`.

```bash
sudo apt update && sudo apt install -y git build-essential python3.12-venv
git clone https://github.com/m-555/explain-to-learn.git ~/explain-to-learn
cd ~/explain-to-learn
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[train,dev]"
```

`build-essential` provides the C compiler that Triton uses to build its GPU kernels on
first use. Installing the training extras keeps the installed PyTorch build, because
the pinned libraries accept 2.14.1. Confirm that `pip show torch` still reports
`2.14.1+cu130`.

### Check Blackwell support

One line checks the PyTorch build, its CUDA version, the GPU's compute capability and
whether the build contains `sm_120` kernels:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(), 'sm_120' in torch.cuda.get_arch_list())"
```

Expected output: `2.14.1+cu130 13.0 (12, 0) True`. A `False` at the end means the
PyTorch build cannot run on the RTX 5090. Reinstall PyTorch from the `cu130` index.

To check that a kernel actually runs on the GPU:

```bash
python -c "import torch; x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); print((x @ x).float().abs().mean().item())"
```

### Run a short training job

Build the SFT data, then train for 20 steps to check memory and speed before a full run:

```bash
python -m explain_to_learn.train_data --silver data/processed/silver_kept.jsonl --out data/processed/sft
python train/train_lora.py --variant r16 --system d --max-steps 20
```

The script prints the peak GPU memory and the run time. See `train/lora_config.yaml` for
the settings and variants.

## 3. Docker alternative

The training image (`docker/Dockerfile.train`) contains PyTorch 2.14.1 for CUDA 13.0 and
the project's training extras. The repository is mounted into the container at run
time, so code changes do not need a rebuild.

### Option A: Docker Desktop

1. Install [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
   with the WSL2 backend (Settings, General, "Use the WSL 2 based engine"). Under
   Settings, Resources, WSL integration, enable the Ubuntu distribution. GPU access
   works through WSL2 with the Windows driver from step 1.
2. Docker supports Docker Desktop only on Windows versions that Microsoft still services.
   If a current Docker Desktop release no longer installs on Windows 10, use option B.

### Option B: Docker Engine inside WSL2

Install Docker Engine in Ubuntu by following the
[Docker Engine instructions for Ubuntu](https://docs.docker.com/engine/install/ubuntu/).
Then install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and register it with Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

`systemctl` needs systemd in WSL2. If it is off, add these lines to `/etc/wsl.conf`,
then run `wsl --shutdown` from Windows:

```ini
[boot]
systemd=true
```

### Build and run

Run these commands from the WSL2 Ubuntu shell, in the repository folder. First check
that containers can use the GPU:

```bash
docker run --rm --gpus all pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime python -c "import torch; print(torch.cuda.get_device_name(0), 'sm_120' in torch.cuda.get_arch_list())"
```

Build the training image and start a run:

```bash
docker build -f docker/Dockerfile.train -t explain-to-learn-train .
```

```bash
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD":/workspace/explain-to-learn \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -e HF_TOKEN -e WANDB_API_KEY \
  explain-to-learn-train \
  python train/train_lora.py --variant r16 --system d
```

`--ipc=host` gives PyTorch's data loader enough shared memory. Mounting the Hugging
Face cache means models are downloaded only once.

## 4. Serving models with vLLM

`scripts/serve_model.sh` starts an OpenAI-compatible server in the official
`vllm/vllm-openai` image. The generation code, the LLM judge and the demo can then call
`http://localhost:8000/v1`. The script needs Docker with GPU access (option A or B).

| Use | Command |
| --- | --- |
| Systems A and B (base model; B only adds the graph snippet to the prompt) | `bash scripts/serve_model.sh Qwen/Qwen3-4B-Instruct-2507` |
| Systems C and D (base model plus LoRA adapters) | `bash scripts/serve_model.sh Qwen/Qwen3-4B-Instruct-2507 system-c=outputs/adapters/c-r16 system-d=outputs/adapters/d-r16` |
| Teacher model for silver data, or the LLM judge | `bash scripts/serve_model.sh <model-id>` |

With adapters, a request picks one by setting `"model": "system-d"`. The base model is
still served under its own name. The teacher model has to fit in 32 GB together with
its KV cache, so for models above about 14B parameters use an FP8 or AWQ checkpoint.
Set `MAX_MODEL_LEN` and `GPU_MEMORY_UTILIZATION` to trade context length for memory.
The script's header lists all settings.

vLLM uses the sampling defaults from the model's `generation_config.json` unless a
request sets them. Pass `temperature` explicitly for reproducible runs.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `no kernel image is available for execution on the device` | PyTorch build without `sm_120` (CUDA 12.6 or older). Reinstall from the `cu130` index. |
| `CUDA driver version is insufficient for CUDA runtime version` | Windows driver older than R580. Update the driver on Windows, then run `wsl --shutdown`. |
| `nvidia-smi: command not found` in WSL2 | WSL2 is out of date or an NVIDIA Linux driver was installed in WSL2. Run `wsl --update` and remove any NVIDIA Linux driver packages. |
| xformers warns that it was built for a different PyTorch | Unsloth's Blackwell guide treats xformers as optional. Run `pip uninstall -y xformers`, or build it for `sm_120` as that guide describes. |
| Triton error after a package upgrade | Each PyTorch release pins its own Triton. Do not upgrade `triton` on its own; reinstall the pinned PyTorch. |
| Out of GPU memory during training | Lower `batch_size` and raise `gradient_accumulation` by the same factor in `train/lora_config.yaml`, so the effective batch stays the same. |

## Sources

- [PyTorch wheel index](https://download.pytorch.org/whl/torch/) and the PyPI metadata of
  `unsloth`, `unsloth-zoo`, `transformers`, `trl`, `peft` and `bitsandbytes`
  (checked for version constraints).
- [Unsloth: fine-tuning on Blackwell and RTX 50-series GPUs](https://unsloth.ai/docs/basics/training-llms-with-blackwell-rtx-50-series-and-unsloth)
- [vLLM: using Docker](https://docs.vllm.ai/en/latest/deployment/docker.html) and
  [LoRA adapters](https://docs.vllm.ai/en/latest/features/lora.html). The image's CUDA version and
  architecture list come from its published configuration (`CUDA_VERSION`, `TORCH_CUDA_ARCH_LIST`).
- [CUDA Toolkit 13.2 release notes](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/)
  (CUDA 13.x needs an R580+ driver).
- [NVIDIA CUDA on WSL user guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

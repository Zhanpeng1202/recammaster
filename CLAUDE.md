# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReCamMaster is a camera-controlled generative video rendering system that re-captures videos with novel camera trajectories. It's built on top of the Wan2.1 text-to-video model and uses DiffSynth-Studio as the underlying framework. The project trains additional camera control modules on the MultiCamVideo Dataset to enable precise camera trajectory manipulation.

## Key Architecture

### Core Pipeline Components

The main pipeline is `WanVideoReCamMasterPipeline` (extends `WanVideoPipeline` from `diffsynth/pipelines/wan_video.py`):
- **Text Encoder**: T5-based encoder for processing text prompts
- **VAE**: Wan2.1 VAE for encoding/decoding videos to/from latent space
- **DiT (Diffusion Transformer)**: The core denoising model with added camera control modules
- **Image Encoder**: Optional encoder for image conditioning

### Camera Control Architecture

Camera control is implemented by adding two modules to each DiT block:
- `cam_encoder`: Linear layer (12→dim) that encodes 3×4 camera pose matrices (flattened)
- `projector`: Linear layer (dim→dim) for projecting encoded camera features

These modules are initialized to zero/identity and trained on paired multi-view videos to learn camera trajectory control.

### Data Flow

1. **Training Data Preparation** (`train_recammaster.py --task data_process`):
   - Loads videos and extracts VAE latents
   - Encodes text prompts using T5
   - Saves preprocessed `.tensors.pth` files alongside videos

2. **Training** (`train_recammaster.py --task train`):
   - Loads pairs of synchronized videos from MultiCamVideo Dataset
   - Concatenates condition video latents with target video latents
   - Computes relative camera poses between condition and target
   - Trains only the camera encoder/projector and self-attention layers
   - Uses MSE loss on predicted noise vs. ground truth

3. **Inference** (`inference_recammaster.py`):
   - Takes source video + target camera trajectory
   - Generates new video following the specified camera path

## Common Development Commands

### Environment Setup
```bash
# Install Rust/Cargo (required for DiffSynth-Studio extensions)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"

# Install the package
pip install -e .

# Install additional training dependencies
pip install lightning pandas websockets
```

### Model Downloads
```bash
# Download Wan2.1 base models
python download_wan2.1.py

# Download ReCamMaster checkpoint manually from HuggingFace
# Place in: models/ReCamMaster/checkpoints/step20000.ckpt
```

### Data Processing (Training)
```bash
# Extract VAE features from MultiCamVideo Dataset
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_recammaster.py \
  --task data_process \
  --dataset_path path/to/MultiCamVideo/Dataset \
  --output_path ./models \
  --text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --tiled \
  --num_frames 81 \
  --height 480 \
  --width 832 \
  --dataloader_num_workers 2
```

### Training
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_recammaster.py \
  --task train \
  --dataset_path recam_train_data \
  --output_path ./models/train \
  --dit_path "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 8000 \
  --max_epochs 100 \
  --learning_rate 1e-4 \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --dataloader_num_workers 4
```

### Inference
```bash
# Test with example data (cam_type 1-10 for different trajectories)
python inference_recammaster.py --cam_type 1

# Test with custom videos
python inference_recammaster.py \
  --cam_type 1 \
  --dataset_path path/to/your/data \
  --ckpt_path path/to/checkpoint.ckpt \
  --output_dir ./results \
  --cfg_scale 5.0
```

### Camera Trajectory Visualization
```bash
python vis_cam.py
```

## Important Implementation Details

### Camera Trajectory Representation

Camera trajectories are stored as 4×4 c2w (camera-to-world) matrices in JSON format. The code:
1. Loads 81 frames of camera data, samples every 4th frame (→21 frames)
2. Converts matrices to c2w format with coordinate system adjustments
3. Computes relative poses between condition camera (first frame) and target trajectory
4. Flattens 3×4 relative pose matrices to 12-dimensional vectors

### Training Data Structure

The MultiCamVideo Dataset structure:
```
dataset/
  train/
    f18_aperture10/  # Different camera parameters
      scene1/
        videos/
          cam01.mp4 - cam10.mp4  # 10 synchronized views
        cameras/
          camera_extrinsics.json  # Camera trajectories
```

Test data requires:
- `videos/` folder with .mp4 files (≥81 frames each)
- `metadata.csv` with columns: file_name, text (video captions)
- `cameras/camera_extrinsics.json` for target trajectories

### Frozen vs Trainable Parameters

During training:
- **Frozen**: VAE, text encoder, most of DiT
- **Trainable**: `cam_encoder`, `projector`, `self_attn` in each DiT block

Training uses PyTorch Lightning with DeepSpeed for distributed training.

### Video Processing

All videos must be:
- At least 81 frames long
- Processed at 480×832 resolution (center crop + resize)
- Normalized to [-1, 1] range

## Camera Trajectory Types

The system supports 10 preset camera trajectories (cam_type 1-10):
1. Pan Right
2. Pan Left
3. Tilt Up
4. Tilt Down
5. Zoom In
6. Zoom Out
7. Translate Up (with rotation)
8. Translate Down (with rotation)
9. Arc Left (with rotation)
10. Arc Right (with rotation)

Custom trajectories can be defined in `example_test_data/cameras/camera_extrinsics.json`.

## Package Structure

- `diffsynth/`: Main package containing models, pipelines, schedulers
  - `models/`: Neural network architectures (DiT, VAE, text encoders)
  - `pipelines/`: Inference pipelines for different model types
  - `schedulers/`: Diffusion schedulers (FlowMatch, etc.)
  - `extensions/`: Additional tools (RIFE, ESRGAN, FastBlend, etc.)
- `train_recammaster.py`: Training script with data processing
- `inference_recammaster.py`: Inference script for generating videos
- `download_wan2.1.py`: Downloads base Wan2.1 models from ModelScope
- `vis_cam.py`: Visualizes camera trajectories

## Dependencies

Core dependencies (from requirements.txt):
- torch>=2.0.0, torchvision
- transformers==4.46.2
- cupy-cuda12x (for CUDA extensions)
- einops (tensor operations)
- imageio (video I/O)
- safetensors (model serialization)
- modelscope (for downloading models)
- sentencepiece, protobuf, ftfy (text processing)

Training adds: lightning, pandas, websockets

# Multimodal RL under Noise — README

> **TL;DR (quick start)**
>
> ```bash
> # 1) Create the conda env
> conda env create -f environment.yml
> conda activate multimodal
>
> # 2) Train a model (example: CORAL encoder + SAC on HalfCheetah, image+depth+state, no extra noise)
> python trainer.py \
>   --seed 0 --algo 6 --rl_algo 0 --env_id 1 \
>   --modalities 3 --no_state 0 --noise_level 0.0 \
>   --render 1 --save_model 1
>
> # 3) Evaluate TODO
> ```
>
> Encodings:
>
> | `--algo` | Encoder / Preprocessor | File |
> |---------:|-------------------------|------|
> | 0 | LinearComb | `preprocessors/linear_comb.py` |
> | 1 | ConCat | `preprocessors/concatenation.py` |
> | 2 | CURL | `preprocessors/curl.py` |
> | 3 | MMM | `preprocessors/mmm.py` |
> | 4 | GMC | `preprocessors/gmc.py` |
> | 5 | AMDF | `preprocessors/amdf.py` |
> | 6 | CORAL | `preprocessors/coral.py` |
>
> | `--rl_algo` | RL Algorithm | File |
> |------------:|--------------|------|
> | 0 | SAC | `rl_algos/sac.py` |
> | 1 | PPO | `rl_algos/ppo.py` |
>
> | `--env_id` | Gymnasium Env | Max Return | Notes |
> |-----------:|----------------|------------|-------|
> | 0 | Ant-v5 | 6000 | |
> | 1 | HalfCheetah-v5 | 10000 | |
> | 2 | Hopper-v5 | 3500 | |
> | 3 | Humanoid-v5 | 6500 | |
> | 4 | Walker2d-v5 | 5500 | |
> | 5 | Pusher-v5 | -20 | |
> | 6 | Reacher-v5 | -3 | |
> | 7 | InvertedPendulum-v5 | 1000 | |
>
> ---

## 1. Project Overview
This repository trains and evaluates reinforcement-learning agents that receive **multimodal observations** (state, RGB image, depth) from MuJoCo control tasks, and exposes them to a variety of **sensor noises**. It lets you:

- Swap in different **representation/encoder modules** (LinearComb, CURL, CORAL, etc.).
- Choose between **SAC** and **PPO**.
- Inject **configurable noise types** (Gaussian, salt & pepper, puzzle patches, hallucinations, etc.) to any modality.
- Log to TensorBoard and save checkpoints.

The core pieces are:

- `trainer_mujoco.py` or `trainer_fetch.py` - trainers for either the mujoco environments of the fetch environments.
- `envs/` — wraps Gymnasium envs, adds multimodal observations + noise injection.
- `preprocessors/` — encoders used to create latent `z` from modalities.
- `rl_algos/` — SAC and PPO implementations that consume the latent representation.
- `configs/` — YAML defaults for RL hyperparameters (`rl.yml`) and per-noise settings (`noises.yml`).
- `utils.py`, `architectures.py`, `rl_utils.py`, `noises.py` — logging, NN blocks, replay buffers, and noise functions.

---

## 2. Environment Setup

### 2.1 Conda (recommended)
```bash
conda env create -f environment.yml
conda activate multimodal
```
This installs Python 3.8 and all pinned pip packages (PyTorch 2.4+, Gymnasium, MuJoCo, etc.). If MuJoCo rendering fails on headless servers, install EGL or set `MUJOCO_GL=egl`.

---

## 4. Running Training
`trainer.py` exposes a minimal CLI (via `argparse`). Important flags:

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `--seed` | int | 0 | Random seed |
| `--algo` | int | 6 | Which *encoder* to use (table above) |
| `--rl_algo` | int | 1 | 0=SAC, 1=PPO |
| `--env_id` | int | 1 | Which Gym env to train on (table above) |
| `--z_dim` | int | 64 | Latent size of representation |
| `--noise_level` | float | 0.0 | Global scalar to scale noise intensity/probability |
| `--render` | int | 1 | Whether to grab frames (1=yes) for logging |
| `--modalities` | int | 3 | How many modalities are used (1=state, 2=+image, 3=+depth) |
| `--no_state` | int | 1 | If 1, removes the privileged low-dim state from inputs |
| `--save_model` | int | 1 | Save checkpoints in `checkpoints/` |
| `--reload` | int | 0 | Resume from latest checkpoint if available |

### 4.1 Example
**SAC + CURL on Walker2d, moderate Gaussian noise, use image+depth only:**
```bash
python trainer.py --seed 42 --algo 2 --rl_algo 0 --env_id 4   --modalities 2 --no_state 1 --noise_level 0.3
```

### 4.2 Changing hyperparameters
- RL hyperparams (batch size, lr, buffer size, etc.) live in `configs/rl.yml`.
- Noise defaults live in `configs/noises.yml`.
- You can edit these YAMLs or load them programmatically inside the scripts.

---

## 5. Noise Configuration
`gym_environment.NoisyEnv` reads `configs/noises.yml` and supports multiple noise functions defined in `noises.py`. Compatibility between modalities and noises is governed by `COMPATIBILITY_NOISES` in `gym_environment.py`.

---

## 7. Logging & Outputs
- **TensorBoard logs:** `logs/<rl_algo>/<exp_name>/` (see `utils.Logger`).
  ```bash
  tensorboard --logdir logs --port 6006
  ```

---

## 9. Tips & Common Pitfalls
- **MuJoCo rendering:** set `export MUJOCO_GL=egl` on servers without a display.
- **Replay buffer size:** adjust `size_buffer` in `configs/rl.yml` to fit GPU memory.
- **Modalities mismatch:** ensure `--modalities` matches what your encoder expects (e.g., some encoders require images).
- **Noise + modality:** Some noises only apply to images or depth; double-check `COMPATIBILITY_NOISES`.
- **Seeds:** SAC uses a random warm-up of `start_steps=1e3` (see `trainer.py`). Make seeds reproducible by setting all torch/np seeds.

---



# Repository Guidelines

## Project Structure

This repository implements an Active Mapping agent using **rsl_rl (PPO)** within a **Legged Gym / Isaac Gym** environment.

* **/envs**: Isaac Gym environment definitions (Patchify environment logic).
* **/learning**: PPO algorithm configurations and custom Actor-Critic architectures.
* **/scripts**: Entry points for training (`train.py`) and visualization (`play.py`).
* **/models**: Saved TorchScript models and checkpoints.
* **/data/GLEAM**: Reference implementation (Baseline for environment data processing).

## Build, Test, and Development Commands

* `python scripts/train.py --task active_mapping`: Start training the drone agent.
* `python scripts/play.py --task active_mapping --load_model [DATE]`: Visualize a trained agent in the GUI.
* `pytest tests/`: Run unit tests for reward calculations and patchification logic.

## Coding Style & Naming Conventions

* **Indentation**: 4 spaces (standard Python PEP 8).
* **Observation Handling**:
* Since we use **flexible numbers of depth images**, observations should be pre-processed.
* Naming: Use `obs_depth_tensor` for raw inputs and `obs_embedding` for encoded features.
* possible encoding strategy might be max pooling directly


* **Environment Constants**: All physical constants (e.g., `FIXED_HEIGHT = 1.0`) must be defined in the `Config` class.

## Testing & Logic Guidelines

### Reward Function (Simplified)

Unlike the complex rewards in GLEAM, this project uses a **Coverage-based Reward**:

* **Primary Reward**: Number of unique patches visited per episode.
* **Penalty**: A large negative reward if `crash == True`.
* **Termination**: Episodes terminate immediately upon collision.

### Observation Space

* Inputs must support a sequence of depth images.
* Refer to `train_rgb_hopper.py` for PPO integration, but extend the `ActorCritic` class to handle `(N, H, W)` depth dimensions.

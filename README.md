# MLCenter

This project explores the implementation of the paper "[Diffusion Policy Policy Optimization](https://arxiv.org/abs/2409.00588)" ([github](https://github.com/irom-princeton/dppo)) using TensorFlow. The aim is to replicate the results presented in the paper, analyze the methodology, and propose improvements or alternative approaches.

## Installation

Please follow the installation guide of from the initial ([repo](https://github.com/irom-princeton/dppo)).

## Usage

### Pre-training

```
# Gym - hopper/walker2d/halfcheetah torch version
python script/run.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/gym/pretrain/hopper-medium-v2

# Gym - hopper/walker2d/halfcheetah tensorflow version
python script/run.py --config-name=pre_diffusion_mlp_tf \
    --config-dir=cfg/gym/pretrain/hopper-medium-v2
```

### Fine-tuning

```
# Gym - hopper/walker2d/halfcheetah torch version
python script/run.py --config-name=ft_ppo_diffusion_mlp \
    --config-dir=cfg/gym/finetune/hopper-v2

# Gym - hopper/walker2d/halfcheetah tensorflow version
python script/run_tf.py --config-name=ft_ppo_diffusion_mlp_tf \
    --config-dir=cfg/gym/finetune/hopper-v2
```

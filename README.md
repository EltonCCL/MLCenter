# MLCenter

This project explores the implementation of the paper "[Diffusion Policy Policy Optimization](https://arxiv.org/abs/2409.00588)" ([github](https://github.com/irom-princeton/dppo)) using TensorFlow. The aim is to replicate the results presented in the paper, analyze the methodology, and propose improvements or alternative approaches.

## Installation

Please follow the installation guide of from the initial ([repo](https://github.com/irom-princeton/dppo)).

fatal error: GL/glew.h: No such file or director
```
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
```
```
export CPATH=$CONDA_PREFIX/include
```

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


## Testing
```
Directory structure:
└── MLCenter (fork from irom-princeton-dppo)
    ├── agent
    │   ├── finetune
    │   │   ├── train_agent_tf.py
    │   │   ├── train_ppo_diffusion_agent_tf.py
    │   │   └── train_ppo_agent_tf.py
    │   ├── dataset
    │   │   └── sequence_tf.py ✓
    │   └── pretrain
    │       ├── train_agent_tf.py ✓
    │       └── train_diffusion_agent_tf.py ✓
    ├── script
    │   ├── run_tf.py
    │   └── run.py
    ├── pyproject.toml
    ├── util
    │   ├── reward_scaling.py
    │   └── scheduler_tf.py ✓
    ├── model
    │   ├── diffusion
    │   │   ├── diffusion_ppo.py 
    │   │   ├── sampling.py ✓
    │   │   ├── mlp_diffusion.py ✓
    │   │   ├── diffusion.py ✓
    │   │   ├── diffusion_vpg.py
    │   │   └── modules.py ✓
    │   └── common
    │       ├── mlp.py ✓
    │       ├── critic.py ✓
    │       └── modules.py ✓
    └── cfg
        └── gym
            ├── finetune
            │   └── hopper-v2
            │       ├── ft_rwr_diffusion_mlp.yaml
            │       ├── ft_ppo_diffusion_mlp.yaml
            │       ├── ft_qsm_diffusion_mlp.yaml
            │       ├── ft_dql_diffusion_mlp.yaml
            │       ├── ft_idql_diffusion_mlp.yaml
            │       ├── ft_ppo_exact_diffusion_mlp.yaml
            │       ├── ft_awr_diffusion_mlp.yaml
            │       └── ft_dipo_diffusion_mlp.yaml
            └── pretrain
                └── hopper-medium-v2
                    └── pre_diffusion_mlp_tf.yaml
```
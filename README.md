# MLCenter

This project explores the implementation of the paper "[Diffusion Policy Policy Optimization](https://arxiv.org/abs/2409.00588)" ([github](https://github.com/irom-princeton/dppo)) using TensorFlow. The aim is to replicate the results presented in the paper, analyze the methodology, and propose improvements or alternative approaches.

## Installation

Note that this code base currently soley focus on the Gym Hopper env only.
You could also check out the installation guide form the initial [repo](https://github.com/irom-princeton/dppo).

1. Clone the repository
```
git@github.com:EltonCCL/MLCenter.git
cd MLCenter
```

2. Install core dependencies with a conda environment (if you do not plan to use Furniture-Bench, a higher Python version such as 3.10 can be installed instead) on a Linux machine with a Nvidia GPU.
```
conda create -n dppo python=3.8 -y
conda activate dppo
pip install -e .
```

3. Install specific Gym environment dependencies
```
pip install -e .[gym]
```

4. Install Tensorflow
```
python3 -m pip install 'tensorflow[and-cuda]'
```

5. Install MujuCo for Gym
```
cd ~
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir $HOME/.mujoco
tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
echo -e 'export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
echo -e 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
echo -e 'export PATH="$LD_LIBRARY_PATH:$PATH"' >> ~/.bashrc
```

6. Set environment variables for data and logging directory (default is data/ and log/), and set WandB entity (username or team name)
```
source script/set_path.sh
```

## Special Issues
fatal error: GL/glew.h: No such file or director
```
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
```
```
export CPATH=$CONDA_PREFIX/include
```

## Checkpoints
You could found the torch checkpoint from the original repo. Here is the pre-trained/fine-tunned weight for the TF version. [OneDrive Link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cceli_connect_ust_hk/Eu76zFJBHZVEmbZ-btp_vZ0B1tcpWd6l8eS2h4QAYBi-kQ?e=bjRBzj).

The OneDrive folder contains the following folders:
- `gym`: containing the weight for pre-training the diffusion mlp in both torch and TF version.
- `hopper-medium-v2_pre_diffusion_mlp_ta4_td20`: pretrained model weight for the diffusion MLP (torch ver)
- `hopper-medium-v2_pre_diffusion_mlp_ta4_td20_tf`: pretrained model weight for the diffusion MLP (TF ver)
- `hopper-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10_tf`: fine-tuned diffusion policy using PPO (TF ver)
- `hopper-medium-v2_sac_diffusion_mlp_ta4_td20_tf_0.1lambda`: fine-tuned diffusion policy using SAC approach (TF ver)
- `hopper-medium-v2_sac_diffusion_mlp_ta4_td20_tf`: fine-tuned diffusion policy using SAC approach (TF ver)

For the TF related script, it contains the checkpoint, and hydra config that used to run the script.

## Basic Usage
### Pre-training

Remember to set the `train_dataset_path` cfg parameter to the path to `train.npz`

```
# Gym - hopper torch version
python script/run.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/gym/pretrain/hopper-medium-v2

# Gym - hopper tensorflow version
python script/run.py --config-name=pre_diffusion_mlp_tf \
    --config-dir=cfg/gym/pretrain/hopper-medium-v2
```

### Fine-tuning

Remember to set the `base_policy_path` cfg parameter to the desired checkpoint. Usually is the pre-trained model checkpoint `hopper-medium-v2_pre_diffusion_mlp_ta4_td20(_tf)`

```
# Gym - hopper torch version (PPO)
python script/run.py --config-name=ft_ppo_diffusion_mlp \
    --config-dir=cfg/gym/finetune/hopper-v2

# Gym - hopper tensorflow version (PPO)
python script/run_tf.py --config-name=ft_ppo_diffusion_mlp_tf \
    --config-dir=cfg/gym/finetune/hopper-v2

# Gym - hopper tensorflow version (SAC)
python script/run_tf.py --config-name=ft_sac_diffusion_mlp_tf \
    --config-dir=cfg/gym/finetune/hopper-v2
```

### Evaluation

Remember to set the `base_policy_path` cfg parameter to the desired checkpoint. It could be the pre-trained checkpoint and fine-tuned checkpoint.

```
# Gym - hopper torch version (PPO)
python script/run.py --config-name=eval_diffusion_mlp \
    --config-dir=cfg/gym/eval/hopper-v2

# Gym - hopper tensorflow version (PPO)
python script/run_tf.py --config-name=eval_ppo_diffusion_mlp_tf \
    --config-dir=cfg/gym/eval/hopper-v2

# Gym - hopper tensorflow version (SAC)
python script/run_tf.py --config-name=eval_sac_diffusion_mlp_tf \
    --config-dir=cfg/gym/eval/hopper-v2
```
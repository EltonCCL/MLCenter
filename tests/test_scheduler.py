import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch
from util.scheduler_tf import CosineAnnealingWarmupRestarts as CosineAnnealingWarmupRestartsTF
from util.scheduler import CosineAnnealingWarmupRestarts
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# [Previous scheduler implementations remain the same]

def test_schedulers(total_steps=1000, save_path='scheduler_comparison.png'):
    # Parameters
    first_cycle_steps = 200
    warmup_steps = 1
    max_lr = 1e-3
    min_lr = 1e-6
    gamma = 1
    cycle_mult = 1
    
    # PyTorch scheduler with AdamW
    model = torch.nn.Linear(10, 1)  # Dummy model to have parameters
    torch_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-6
    )
    torch_scheduler = CosineAnnealingWarmupRestarts(
        optimizer=torch_optimizer,
        first_cycle_steps=first_cycle_steps,
        cycle_mult=cycle_mult,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        gamma=gamma
    )

    # TensorFlow scheduler
    tf_scheduler = CosineAnnealingWarmupRestartsTF(
        first_cycle_steps=first_cycle_steps,
        cycle_mult=cycle_mult,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        gamma=gamma
    )

    # Collect learning rates
    torch_lrs = []
    tf_lrs = []

    for step in range(total_steps):
        # PyTorch
        torch_lr = torch_scheduler.get_lr()[0]
        torch_lrs.append(torch_lr)
        torch_scheduler.step()

        # TensorFlow
        tf_lr = tf_scheduler(tf.cast(step, tf.float32)).numpy()
        tf_lrs.append(tf_lr)

    torch_lrs = np.array(torch_lrs)
    tf_lrs = np.array(tf_lrs)
    
    # Calculate differences
    abs_diff = np.abs(torch_lrs - tf_lrs)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    print(f"Maximum absolute difference: {max_diff:.8f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")

    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Plot learning rates
    plt.subplot(2, 1, 1)
    plt.plot(torch_lrs, label='PyTorch', alpha=0.7)
    plt.plot(tf_lrs, label='TensorFlow', linestyle='--', alpha=0.7)
    plt.title('Learning Rate Schedules Comparison (AdamW)')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.legend()
    
    # Plot absolute difference
    plt.subplot(2, 1, 2)
    plt.plot(abs_diff, label='|PyTorch - TensorFlow|')
    plt.title('Absolute Difference Between Implementations')
    plt.xlabel('Step')
    plt.ylabel('Absolute Difference')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    # Show the plot
    plt.show()

    assert max_diff < 0.001, f"Maximum absolute difference is too high: {max_diff:.8f}"

    # return torch_lrs, tf_lrs

# Run the test and save the plot
if __name__ == '__main__':
    torch_lrs, tf_lrs = test_schedulers()
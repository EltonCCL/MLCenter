import tensorflow as tf
import numpy as np
import math

class CosineAnnealingWarmupRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine Annealing Learning Rate Scheduler with Warmup and Restarts.

    Args:
        first_cycle_steps (int): Number of steps for the first cycle.
        cycle_mult (float, optional): Factor to increase cycle steps. Defaults to 1.0.
        max_lr (float, optional): Maximum learning rate. Defaults to 0.1.
        min_lr (float, optional): Minimum learning rate. Defaults to 0.001.
        warmup_steps (int, optional): Number of warmup steps. Defaults to 0.
        gamma (float, optional): Decay rate for learning rate after each cycle. Defaults to 1.0.
    """
    def __init__(
        self,
        first_cycle_steps,
        cycle_mult=1.0,
        max_lr=0.1,
        min_lr=0.001,
        warmup_steps=0,
        gamma=1.0,
        n_critic_warmup_itr = 0,
    ):
        super().__init__()
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.n_critic_warmup_itr = n_critic_warmup_itr

    def __call__(self, step):
        """Calculate the learning rate at a given step.

        Args:
            step (int): The current training step.

        Returns:
            tf.Tensor: The learning rate for the current step.
        """
        step = tf.cast(step, tf.float32)

        effective_step = tf.maximum(step - self.n_critic_warmup_itr, 0)
        
        if self.cycle_mult == 1.0:
            # Simple case: fixed cycle length
            cycle = tf.floor(effective_step / self.first_cycle_steps)
            step_in_cycle = effective_step % self.first_cycle_steps
            current_cycle_steps = self.first_cycle_steps
        else:
            # Complex case: increasing cycle length
            n = tf.math.log(
                effective_step / self.first_cycle_steps * (self.cycle_mult - 1) + 1
            ) / tf.math.log(self.cycle_mult)
            n = tf.floor(n)
            
            cycle = n
            step_in_cycle = effective_step - self.first_cycle_steps * (
                tf.pow(self.cycle_mult, n) - 1
            ) / (self.cycle_mult - 1)
            current_cycle_steps = self.first_cycle_steps * tf.pow(self.cycle_mult, n)
        
        # Warmup phase
        if_warmup = tf.cast(step_in_cycle < self.warmup_steps, tf.float32)
        warmup_lr = self.min_lr + (self.max_lr - self.min_lr) * (
            step_in_cycle / self.warmup_steps
        )
        
        # Cosine annealing phase
        cosine_lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
            1 + tf.cos(
                tf.constant(math.pi) * 
                (step_in_cycle - self.warmup_steps) / 
                (current_cycle_steps - self.warmup_steps)
            )
        )
        
        # Select appropriate learning rate based on phase
        lr = tf.where(step_in_cycle < self.warmup_steps, warmup_lr, cosine_lr)
        
        # Apply gamma decay
        lr = lr * tf.pow(self.gamma, cycle)
        
        return lr

    def get_config(self):
        """Return the configuration of the learning rate schedule.

        Returns:
            dict: Configuration dictionary.
        """
        return {
            "first_cycle_steps": self.first_cycle_steps,
            "cycle_mult": self.cycle_mult,
            "max_lr": self.base_max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "gamma": self.gamma,
            "n_critic_warmup_itr": self.n_critic_warmup_itr
        }
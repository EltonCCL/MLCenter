import tensorflow as tf
import numpy as np
import math

class CosineAnnealingWarmupRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        first_cycle_steps,
        cycle_mult=1.0,
        max_lr=0.1,
        min_lr=0.001,
        warmup_steps=0,
        gamma=1.0
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

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        if self.cycle_mult == 1.0:
            # Simple case: fixed cycle length
            cycle = tf.floor(step / self.first_cycle_steps)
            step_in_cycle = step % self.first_cycle_steps
            current_cycle_steps = self.first_cycle_steps
        else:
            # Complex case: increasing cycle length
            n = tf.math.log(
                step / self.first_cycle_steps * (self.cycle_mult - 1) + 1
            ) / tf.math.log(self.cycle_mult)
            n = tf.floor(n)
            
            cycle = n
            step_in_cycle = step - self.first_cycle_steps * (
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
        return {
            "first_cycle_steps": self.first_cycle_steps,
            "cycle_mult": self.cycle_mult,
            "max_lr": self.base_max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "gamma": self.gamma
        }
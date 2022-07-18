from .config import CfgNode as CN

__all__ = ['exponential_decay', 'reduce_on_plateau']

# exponential decay
exponential_decay = CN()
exponential_decay.decay_steps = 1e4
exponential_decay.decay_rate = 0.1
exponential_decay.staircase = True

# reduce on plateau
reduce_on_plateau = CN()
reduce_on_plateau.factor = 0.5
reduce_on_plateau.patience = 5
reduce_on_plateau.min_delta = 1e-3
reduce_on_plateau.min_lr = 1e-6
reduce_on_plateau.cooldown = 2

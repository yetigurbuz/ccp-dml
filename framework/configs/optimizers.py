from .config import CfgNode as CN

__all__ = ['SGD', 'RMSprop', 'Adam', 'AdamLRM']

SGD = CN()
SGD.momentum = 0.0
SGD.nesterov = False

RMSprop = CN()
RMSprop.rho = 0.9
RMSprop.momentum = 0.0

Adam = CN()
Adam.beta_1 = 0.9
Adam.beta_2 = 0.999

AdamLRM = CN()
AdamLRM.beta_1 = 0.9
AdamLRM.beta_2 = 0.999
AdamLRM.lr_multiplier = CN()
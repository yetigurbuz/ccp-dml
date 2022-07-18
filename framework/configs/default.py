from .config import CfgNode as CN

from . import optimizers
from . import learning_rate_schedulers


cfg = CN()

cfg.verbose = 2

cfg.root = './configs'

# model configs
# ==============
model = CN()
model.name = ''
model.embedding_head = CN()
model.embedding_head.embedding_size = 128
model.embedding_head.arch = 'GlobalPooling'
model.backbone = CN()
model.backbone.arch = 'BNInception'
model.num_models = 4

# loss configs
# =============
loss = CN()
loss.function = 'contrastive'
loss.computation_head = CN()
loss.computation_head.normalize_embeddings = True
loss.computation_head.lipschitz_cont = False
loss.computation_head.distance_function = 'l2' # l2 or cos
loss.computation_head.squared_distance = False
loss.computation_head.miner = None
loss.computation_head.avg_nonzero_only = True
loss.computation_head.use_proxy = False
loss.computation_head.proxy_anchor = False  # true: anchors are always proxies; false: anchors are always samples
loss.computation_head.use_intra_proxy_pairs = False # true: compute ||proxy_i - proxy_j|| based loss as well


# optimizer configs
# ==================
optimizer = CN()
optimizer.method = 'Adam'
optimizer.learning_rate = 1e-5
optimizer.warm_up_learning_rate_multiplier = 1.0
optimizer.learning_rate_scheduler = CN()
optimizer.learning_rate_scheduler.method = 'reduce_on_plateau'   # or exponential_decay
optimizer.gradient_transformers = CN()
optimizer.gradient_transformers.weight_decay = 1e-4
optimizer.gradient_transformers.clipnorm = 10.0
optimizer.gradient_transformers.clipvalue = 10.0
optimizer.gradient_transformers.excluded_vars = ['bn', 'batchnorm']

# schedulers
for sc in learning_rate_schedulers.__all__:
    optimizer.learning_rate_scheduler.__setattr__(sc, getattr(learning_rate_schedulers, sc))

# optimizers
for opt in optimizers.__all__:
    optimizer.__setattr__(opt, getattr(optimizers, opt))

# training configs
# =================
training = CN()
training.classes_per_batch = 8
training.sample_per_class = 4
training.steps_per_epoch = 100
training.max_epochs = 10000
training.early_stopping_patience = 15
training.min_improvement_margin = 1e-5
training.warm_start = 0
training.freeze_during_warmup = True    # whether freeze the backbone model during warm up epochs
training.exclude_freeze = ['EmbeddingHead']
training.exclude_defrost = []
training.output_dir = '../training/metric_learning'

# validation configs
# ===================
validation = CN()
validation.metric = 'DMLEval'
validation.batch_size = 32

# dataset configs
# ===================
dataset = CN()
dataset.name = ''
dataset.root_path = '../datasets'
dataset.preprocessing = CN()
dataset.preprocessing.method = 'DMLPreprocessing'

# merge all configuration
# ========================
cfg.model = model
cfg.loss = loss
cfg.optimizer = optimizer
cfg.training = training
cfg.validation = validation
cfg.dataset = dataset



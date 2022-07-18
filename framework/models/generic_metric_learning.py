import tensorflow as tf
from . import BaseModel
from . import backbones
from ..layers import pooling as heads

def genericEmbeddingModel(cfg, name='EmbeddingModel', **kwargs):

    # input layer
    w, h = cfg.dataset.preprocessing.get(cfg.dataset.preprocessing.method).out_image_size
    c = cfg.dataset.preprocessing.get(cfg.dataset.preprocessing.method).out_image_channels

    inputs = tf.keras.Input(shape=(w, h, c), name='Input')

    # backbone model
    backbone_arch = cfg.model.backbone.arch
    backbone_cfg = cfg.model.backbone.get(backbone_arch)
    backbone = getattr(backbones, backbone_arch)(**backbone_cfg.arch_parameters, input_shape=inputs.shape,
                                                 name=cfg.model.backbone.arch)

    # embedding head
    embedding_arch = cfg.model.embedding_head.arch
    embedding_cfg = cfg.model.embedding_head.get(embedding_arch)
    embedding = getattr(heads, embedding_arch)(embedding_size=cfg.model.embedding_head.embedding_size,
                                               name='EmbeddingHead',
                                               **embedding_cfg,
                                               )

    # constructing the model
    feat = backbone(inputs)
    emb = embedding(feat)
    name = backbone.name + '_{}'.format(name)

    model = BaseModel(inputs=inputs, outputs=emb, name=name)

    return model



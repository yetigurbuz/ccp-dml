#!/usr/bin/env python
# coding: utf-8
from .contrastive_loss import contrastive, original_contrastive
from .margin_loss import MarginLoss
from .multi_similarity_loss import multi_similarity
from .triplet_loss import triplet
from .xent_loss import XEntLoss
from .proxy_anchor_loss import proxy_anchor
from .proxy_nca_loss import proxy_nca
from .soft_triple_loss import soft_triple
from .mdr import MDRLoss
from .xbm import XBMLoss as xbm
from .proxy_synthesis import PSLoss as ps
from .virtual_softmax import VirtualSoftmaxLoss as virtual_softmax

from .ccp import ccp

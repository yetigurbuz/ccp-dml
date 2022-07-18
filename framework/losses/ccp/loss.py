import tensorflow as tf
import math

from ...datasets import samplers
from ... import datasets
from ...utilities import proxy_utils, dataset_utils
from ...solvers.gradient_transformers import DistanceRegularizer
from ...layers import L2Normalization
from ...utilities.loss_and_metric_utils import labels2Pairs

from ... import losses

from .callback import AlternatingProjectionCallback

from ...configs import default
from ...configs.config import CfgNode as CN


# chance constrained programming (ccp) configs
# ==============================================
ccp = CN()
ccp.function = 'contrastive'
ccp.regularization_coeff = 2.0e-4
ccp.coeff_decay = 1.0
ccp.perturb_noise = 0.0
ccp.min_iterations = 1
ccp.step_at_batch = False
ccp.early_stopping = True
ccp.early_stopping_patience = 3
ccp.monitor_to_stop = 'MAPatR'
ccp.stop_at_min = False
ccp.use_representative_proxy = True
ccp.proxy_per_class = 7
ccp.representative_pool_size = 12
ccp.proxy_sampling_batch_size = 32
ccp.random_init = False
ccp.proxy_lrm = 1.0
ccp.preprocess_representative = True
ccp.normalized_embeddings = False
ccp.ccp_weight = 1.0
ccp.loss_weight = 0.1

default.cfg.loss.ccp = ccp

class CCPLoss(tf.keras.losses.Loss):

    def __init__(self, model, cfg, **kwargs):

        if 'name' in kwargs.keys():
            name = kwargs['name']
        else:
            name = 'ccp'

        super(CCPLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)

        loss_class = getattr(losses, cfg.loss.ccp.function)
        loss_params = cfg.loss.get(cfg.loss.ccp.function)

        class LossWithCCP(loss_class):
            def __init__(self, model, cfg, **kwargs):
                self.profs_weight = cfg.loss.ccp.ccp_weight
                self.pair_loss_weight = cfg.loss.ccp.loss_weight

                super(LossWithCCP, self).__init__(model=model, cfg=cfg, **kwargs)

            def call(self, y_true, y_pred):
                labels, embeddings = y_true, y_pred

                if self.normalize_embeddings:
                    embeddings = L2Normalization()(embeddings)

                pdists = self.pdist_fn(embeddings=embeddings,
                                       ref_embeddings=embeddings,
                                       squared=self.squared_distance)
                pos_pair_mask, neg_pair_mask = labels2Pairs(labels=labels,
                                                            ref_labels=None,
                                                            structured=self.structured_batch,
                                                            num_classes=self.classes_per_batch)

                loss0 = self._computeLoss(pdists, pos_pair_mask, neg_pair_mask, labels)

                profs_loss = super(LossWithCCP, self).call(y_true, y_pred)

                return self.pair_loss_weight * loss0 + self.profs_weight * profs_loss

        ccp_lambda = cfg.loss.ccp.regularization_coeff
        lambda_decay = cfg.loss.ccp.coeff_decay
        min_ccp_iterations = cfg.loss.ccp.min_iterations

        # ccp based optimization
        update_callback = None

        model_weights = model.arch.trainable_weights
        ccp_regularizer = DistanceRegularizer(model_weights=model_weights, penalty_weight=ccp_lambda)

        class_representatives = [None]
        representative_sampler = None
        if cfg.loss.ccp.use_representative_proxy:
            dataset = getattr(datasets, cfg.dataset.name)(dataset_dir=cfg.dataset.root_path, verbose=0)

            one_per_class = samplers.OnePerClass(batch_size=cfg.loss.ccp.proxy_sampling_batch_size)
            preprocessing = getattr(dataset_utils,
                                    cfg.dataset.preprocessing.method)(
                mean_image=dataset.mean,
                std_image=dataset.std,
                **cfg.model.backbone.get(cfg.model.backbone.arch).input_parameters,
                **cfg.dataset.preprocessing.get(cfg.dataset.preprocessing.method))
            representative_dataset = dataset.makeBatch(subset='train',
                                                       split_id=model.ensemble_id,
                                                       num_splits=cfg.model.num_models,
                                                       sampling_fn=one_per_class,
                                                       preprocess_fn=preprocessing
                                                       )

            one_per_class_sampler = iter(representative_dataset.map(lambda *x: x[0]))  # ignore labels

            num_classes = dataset.num_classes.train_split[model.ensemble_id - 1]
            iterations_per_span = math.ceil(num_classes / cfg.loss.ccp.proxy_sampling_batch_size)

            class_representatives, _ = proxy_utils.makeProxy(model=model.arch,
                                                             num_classes=num_classes,
                                                             proxy_per_class=cfg.loss.ccp.proxy_per_class,
                                                             num_sets=1,
                                                             trainable=True,
                                                             name='ccp')

            embedding_layer = model.arch.get_layer('EmbeddingHead')
            embedding_size = embedding_layer._embedding_size
            embedding_model = tf.keras.Model(inputs=model.arch.input, outputs=embedding_layer.output)


            @tf.function
            def representative_sampler():
                k_0 = tf.constant(0, dtype=tf.int32)
                s_0 = tf.TensorArray(dtype=tf.float32, size=iterations_per_span, dynamic_size=False,
                                     clear_after_read=False,
                                     element_shape=tf.TensorShape([None, embedding_size]))

                s, k = tf.while_loop(
                    cond=lambda s_in, k_in: tf.less(k_in, iterations_per_span),
                    body=lambda s_in, k_in: (s_in.write(k_in,
                                                        value=embedding_model(next(one_per_class_sampler),
                                                                              training=False)[:, :embedding_size]),
                                             tf.add(k_in, 1)),
                    loop_vars=(s_0, k_0))

                return tf.reshape(s.concat(), shape=(num_classes, embedding_size))

        if min_ccp_iterations > 0:
            update_callback = AlternatingProjectionCallback(curr_weights=ccp_regularizer._prev_weights,
                                                            model_weigths=model_weights,
                                                            ccp_lambda=ccp_regularizer._lambda,
                                                            lambda_decay=lambda_decay,
                                                            ccp_iterations=min_ccp_iterations,
                                                            perturb_noise=cfg.loss.ccp.perturb_noise,
                                                            step_at_batch=cfg.loss.ccp.step_at_batch,
                                                            early_stopping=cfg.loss.ccp.early_stopping,
                                                            early_stopping_patience=cfg.loss.ccp.early_stopping_patience,
                                                            monitor_to_stop=cfg.loss.ccp.monitor_to_stop,
                                                            stop_at_min=cfg.loss.ccp.stop_at_min,
                                                            class_representatives=class_representatives[0],
                                                            proxy_per_class=cfg.loss.ccp.proxy_per_class,
                                                            representative_sampler=representative_sampler,
                                                            representative_pool_size=cfg.loss.ccp.representative_pool_size,
                                                            random_init=cfg.loss.ccp.random_init,
                                                            normalized_embeddings=cfg.loss.ccp.normalized_embeddings)

        self.loss_function = LossWithCCP(model, cfg, **kwargs)

        if update_callback is not None:
            model.training_callbacks.append(update_callback)

        model.gradient_transformers.append(ccp_regularizer)

    def call(self, y_true, y_pred):
        return self.loss_function.call(y_true, y_pred)


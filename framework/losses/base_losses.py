import tensorflow as tf
from ..utilities.loss_and_metric_utils import getPairwiseDistances, getPairwiseDistancesAutoDiff, neg_cosine_similarity
from ..utilities.loss_and_metric_utils import labels2Pairs
from ..layers import L2Normalization
from . import miners
from ..utilities import proxy_utils

class GenericPairBasedLoss(tf.keras.losses.Loss):
    def __init__(self,
                 model=None,
                 normalize_embeddings=False,
                 lipschitz_cont=True,
                 squared_distance=False,
                 avg_nonzero_only=True,
                 miner=None,
                 distance_function='l2',
                 weight=1.,
                 structured_batch=False,
                 classes_per_batch=None,
                 use_ref_samples=False,
                 use_intra_proxy_pairs=False,
                 use_proxy=False,
                 proxy_anchor=True,
                 num_classes=None,
                 proxy_per_class=1,
                 proxy_name=None,
                 proxy_lrm=1.0,
                 name=None, **kwargs):
        super(GenericPairBasedLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)

        if not isinstance(model.arch, tf.keras.models.Model):
            raise ValueError('Expected {} as model.arch but got {}'.format(tf.keras.models.Model, model))

        self.normalize_embeddings = normalize_embeddings
        self.squared_distance = squared_distance

        if lipschitz_cont:
            print('\033[3;34m' + 'INFO:Loss:Normalization: ' + 'Lipschitz cont. L2 Normalization is used.' + '\033[0m')
            self.l2normalization = L2Normalization()
        else:
            print('\033[3;34m' + 'INFO:Loss:Normalization: ' + 'Classical L2 Normalization is used.' + '\033[0m')
            self.l2normalization = lambda x: tf.nn.l2_normalize(x, axis=-1)

        self.structured_batch = structured_batch
        if structured_batch and (classes_per_batch is None):
            raise ValueError('classes_per_batch should be provided for structured_batch=True configuration.')
        self.use_ref_samples = use_ref_samples
        if use_ref_samples and (classes_per_batch is None):
            raise ValueError('classes_per_batch should be provided for use_ref_samples=True configuration.')
        self.classes_per_batch = classes_per_batch

        self.avg_nonzero_only = avg_nonzero_only
        self.loss_weight = weight
        self.miner = miner

        self.use_proxy = use_proxy
        self._initProxySetting(model.arch, num_classes, proxy_per_class, proxy_name)
        model.learning_rate_multipliers.update({'{}/proxy_embedding'.format(proxy_name): proxy_lrm})

        self.proxy_anchor = proxy_anchor

        if self.use_ref_samples:
            self.structured_batch = False

        self.use_intra_proxy_pairs = use_intra_proxy_pairs

        self.pdist_fn = getPairwiseDistances
        if isinstance(distance_function, str):
            if distance_function.lower() == 'l2':
                print('\033[3;34m' + 'INFO:Loss:Distance: ' + 'L2 distance with power {} is used.'.format(2 if self.squared_distance else 1) + '\033[0m')
                self.pdist_fn = getPairwiseDistances
            elif distance_function.lower() == 'cos':
                print('\033[3;34m' + 'INFO:Loss:Distance: ' + 'Negative Cosine similarity is used.' + '\033[0m')

                self.pdist_fn = neg_cosine_similarity

        self.global_step = tf.Variable(initial_value=0,
                                       trainable=False, shape=(), dtype=tf.int64)


    def _initProxySetting(self, model, num_classes, proxy_per_class, proxy_name):

        self.num_classes = num_classes
        self.proxy_per_class = proxy_per_class

        '''self.class_representatives = next((w for w in model.weights
                                           if 'proxy' in w.name),
                                          None)'''

        self.class_representatives = [w for w in model.weights if 'proxy_embedding' in w.name]

        self.proxy_labels = [w for w in model.weights if 'proxy_label' in w.name]

        # check whether representatives already exist
        if len(self.class_representatives):

            self.use_proxy = True
            num_proxy = sum([p.get_shape().as_list()[0] for p in self.class_representatives])
            if self.num_classes is not None:
                self.proxy_per_class = num_proxy // self.num_classes
            else:
                self.proxy_per_class = 1

            num_label = sum([l.get_shape().as_list()[0] for l in self.proxy_labels])

            if (num_label != num_proxy) or (len(self.class_representatives) != len(self.proxy_labels)):
                raise ValueError('\033[3;31m' +
                                 'Proxy labels are not consistent with Proxies! Revise Proxy declaration steps and ' +
                                 'make sure you use proxy_utils.makeProxy function to be on the safe side.' +
                                 '\033[0m')

        elif self.use_proxy:   # if not exists then define class representatives
            proxy_name = 'class' if proxy_name is None else proxy_name
            self.class_representatives, self.proxy_labels = proxy_utils.makeProxy(
                model=model,
                num_classes=self.num_classes,
                proxy_per_class=self.proxy_per_class,
                num_sets=1,
                name=proxy_name,
                trainable=True)

            '''embedding_size = model.get_layer('embedding').output_shape[-1]
            num_proxy = self.num_classes * self.proxy_per_class
            self.class_representatives = [model.add_weight(name='class/proxy_embedding',
                                                           shape=(num_proxy, embedding_size),
                                                           dtype=tf.float32,
                                                           initializer=tf.keras.initializers.glorot_uniform,
                                                           trainable=True)]
            self.proxy_labels = [model.add_weight(name='class/proxy_label',
                                                  shape=(num_proxy, 1),
                                                  dtype=tf.int32,
                                                  initializer=tf.keras.initializers.glorot_uniform,
                                                  trainable=False)]'''


        if self.use_proxy:
            '''self.proxy_labels = tf.expand_dims(
                tf.tile(tf.range(self.num_classes),
                        multiples=[self.proxy_per_class]),
                axis=1)'''
            self.use_ref_samples = True
            self.structured_batch = False

    def call(self, y_true, y_pred):

        labels, embeddings = y_true, y_pred

        if self.normalize_embeddings:
            embeddings = self.l2normalization(embeddings)

        if self.use_ref_samples:
            if self.use_proxy:
                ref_embeddings = tf.concat(self.class_representatives, axis=0)
                ref_labels = tf.concat(self.proxy_labels, axis=0)

                # required for xbm
                if len(ref_embeddings.shape.as_list()) == 3:
                    d1, d2, d3 = ref_embeddings.shape.as_list()
                    ref_embeddings = tf.reshape(ref_embeddings, shape=(d1 * d2, d3))
                    ref_labels = tf.reshape(ref_labels, shape=(d1 * d2, 1))

                if self.normalize_embeddings:
                    ref_embeddings = self.l2normalization(ref_embeddings)

                if self.use_intra_proxy_pairs:
                    embeddings = tf.concat([ref_embeddings, embeddings], axis=0)
                    labels = tf.concat([ref_labels, labels], axis=0)
            else:
                ref_embeddings = embeddings[:self.classes_per_batch]
                ref_labels = labels[:self.classes_per_batch]
                if not self.use_intra_proxy_pairs:
                    embeddings = embeddings[self.classes_per_batch:]
                    labels = labels[self.classes_per_batch:]
            ref_labels_to_pass = ref_labels
        else:
            ref_embeddings = None
            ref_labels = None
            ref_labels_to_pass = labels

        pdists = self.pdist_fn(embeddings=embeddings,
                               ref_embeddings=ref_embeddings,
                               squared=self.squared_distance)

        pos_pair_mask, neg_pair_mask = labels2Pairs(labels=labels,
                                                    ref_labels=ref_labels,
                                                    structured=self.structured_batch,
                                                    num_classes=self.classes_per_batch)

        if not self.proxy_anchor: # then samples are anchors rather than proxies
            pdists = tf.transpose(pdists)
            pos_pair_mask = tf.transpose(pos_pair_mask)
            neg_pair_mask = tf.transpose(neg_pair_mask)
            ref_labels_to_pass = labels

        if self.miner is not None:

            miner_fn = getattr(miners, self.miner)
            pos_pair_mask, neg_pair_mask = miner_fn(pdists=pdists,
                                                    pos_pair_mask=pos_pair_mask,
                                                    neg_pair_mask=neg_pair_mask,
                                                    wrt_representatives=self.use_ref_samples)

        pos_pair_mask = tf.stop_gradient(pos_pair_mask)
        neg_pair_mask = tf.stop_gradient(neg_pair_mask)

        loss = self._computeLoss(pdists, pos_pair_mask, neg_pair_mask, ref_labels_to_pass)

        # summary ===============================
        self.logPDistSummary(embeddings, labels)

        return self.loss_weight * loss

    def _computeLoss(self, pdists, pos_pair_mask, neg_pair_mask, ref_labels):
        """
        This has to be implemented and is what actually computes the loss.
        """
        raise NotImplementedError

    def logPDistSummary(self, embeddings, labels):
        sample_dists = self.pdist_fn(embeddings=embeddings,
                                     squared=self.squared_distance)
        pos_pairs, neg_pairs = labels2Pairs(labels=labels,
                                            structured=self.structured_batch,
                                            num_classes=self.classes_per_batch)
        pos_dists = sample_dists[tf.cast(pos_pairs, tf.bool)]
        neg_dists = sample_dists[tf.cast(neg_pairs, tf.bool)]

        tf.summary.histogram('pos_pdists', pos_dists, step=self.global_step)
        tf.summary.histogram('neg_pdists', neg_dists, step=self.global_step)

        self.global_step.assign_add(tf.constant(1, dtype=tf.int64))
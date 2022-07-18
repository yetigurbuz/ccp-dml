import numpy as np
import tensorflow as tf

from sklearn.decomposition import PCA

import os
import shutil

from .. import datasets
from .. import losses
from .. import metrics
from .. import models
from .. import layers
from .. import utilities
from .gradient_transformers import updateGradientTransformers
from .optimizers import Optimizers

class DMLFramework(object):

    def __init__(self, cfg):

        self.cfg = cfg

        self.model_name = self.getModelName()

        suffix = ''
        if len(cfg.model.name):
            model_name = cfg.model.name
            if '-' in model_name:
                model_name, suffix = model_name.split('-', 1)
                suffix = '-' + suffix
            self.model_name = model_name

            self.saved_models_dir = os.path.join(cfg.training.output_dir,
                                                 self.model_name)

            self.model_save_dirs = []
            if 'Ensemble' in self.model_name:
                num_models = int(self.model_name.strip('_Model').split('_')[-1])
                self.model_save_dirs = [
                    os.path.join(self.saved_models_dir, 'model_{}'.format(m + 1))
                    for m in range(num_models)]
            else:
                self.model_save_dirs = [self.saved_models_dir]

        else:
            self.saved_models_dir = os.path.join(cfg.training.output_dir,
                                                 self.model_name)
            os.makedirs(self.saved_models_dir, exist_ok=True)

            self.model_save_dirs = []

            if cfg.model.num_models > 1:
                self.model_save_dirs = [
                    os.path.join(self.saved_models_dir, 'model_{}'.format(m+1))
                    for m in range(cfg.model.num_models)]
                [os.makedirs(dir, exist_ok=True) for dir in self.model_save_dirs]
            else:
                self.model_save_dirs = [self.saved_models_dir]

        self.trained_models = self.getTrainedModels(suffix)


    def getModelName(self):

        # ================================================================================================
        proxy_text = 'Proxy_' if bool(self.cfg.loss.computation_head.use_proxy) else ''
        loss_text = ''.join([s.capitalize() for s in self.cfg.loss.function.split('_')])
        sub_loss_text = \
            self.cfg.loss.get(self.cfg.loss.function).function \
                if hasattr(self.cfg.loss.get(self.cfg.loss.function), 'function') else ''
        sub_loss_text = ''.join([s.capitalize() for s in sub_loss_text.split('_')])
        loss_text = loss_text.upper() + '_{}Loss_'.format(sub_loss_text) if len(sub_loss_text) else loss_text + 'Loss_'

        backbone_text = self.cfg.model.backbone.arch + '_'
        embedding_head_text = \
            self.cfg.model.embedding_head.arch + \
            '_{}_'.format(self.cfg.model.embedding_head.embedding_size)
        # ''.join([s.capitalize() for s in self.cfg.model.embedding_head.arch.split('_')]) + \

        ensemble_text = 'Ensemble_{}_Model'.format(self.cfg.model.num_models) \
            if self.cfg.model.num_models > 1 else 'Single_Model'
        batch_text = '{}x{}_'.format(self.cfg.training.classes_per_batch,
                                     self.cfg.training.sample_per_class)
        dataset_text = self.cfg.dataset.name + '_Dataset_'

        model_identifier = ''.join([backbone_text, embedding_head_text, proxy_text,
                                    loss_text, dataset_text, batch_text, ensemble_text])

        return model_identifier

    def getTrainedModels(self, suffix=''):

        all_models = []
        for dir in self.model_save_dirs:
            model_files = []
            for file in sorted(os.listdir(dir)):
                if file.startswith(self.model_name+suffix) and file.endswith('h5'):
                    model_files.append(os.path.join(dir, file))
            all_models.append(model_files)

        return all_models

    def clearTrainedModels(self):

        for training_dir in self.model_save_dirs:
            for fname in os.listdir(training_dir):
                file_path = os.path.join(training_dir, fname)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        for k in range(len(self.trained_models)):
            self.trained_models[k] = []



    def freezeModel(self, model, excluded=['EmbeddingHead']):

        for layer in model.layers:
            if hasattr(layer, 'layers'):
                self.freezeModel(layer, excluded)

            elif not any([((e in str(layer.__class__)) or (e in str(layer.name))) for e in excluded]):
                layer.trainable = False
            else:
                print('found {} layers to be excluded in freeze'.format(excluded))

    def defrostModel(self, model, excluded=['BatchNormalization']):

        for layer in model.layers:

            if hasattr(layer, 'layers'):
                self.defrostModel(layer, excluded)

            elif not any([((e in str(layer.__class__)) or (e in str(layer.name))) for e in excluded]):
                layer.trainable = True
            else:
                print('found {} layers to be excluded in defrost'.format(excluded))

    def logTraining(self, score, **hyperparams):

        params = self.cfg.clone()

        # set hyperparameters
        if self.cfg.verbose and len(hyperparams):
            print('Hyperparameters:')
        for key, val in hyperparams.items():
            subkeys = key.split('/')
            hyp = params
            for subkey in subkeys[:-1]:
                hyp = hyp.get(subkey)
            hyp.__setattr__(subkeys[-1], val)

        loss_id = self.cfg.loss.function
        optimizer_id = self.cfg.optimizer.method

        top_line = '\n=== training configuration ===\n'
        bottom_line = '=== end of training configuration ===\n'
        training_line = 'optimizer: {}, learning_rate: {}, weight_decay: {}\n'.format(
            optimizer_id,
            self.cfg.optimizer.learning_rate,
            self.cfg.optimizer.gradient_transformers.weight_decay)

        loss_common_line = 'losses:common: {}\n'.format(self.cfg.loss.computation_head)
        loss_line = 'losses:{}: {}\n'.format(loss_id, self.cfg.loss.get(loss_id))
        score_line = 'score: {}\n'.format(score)

        fpath = os.path.join(self.saved_models_dir, '{}_Training_Logs.txt'.format(self.model_name))
        with open(fpath, 'a') as f:
            f.writelines([top_line, training_line, loss_common_line, loss_line, score_line, bottom_line])

    def logSingleTraining(self, score, model_id):

        top_line = '\n=== model id: {} training log ===\n'.format(model_id)
        bottom_line = '=== end of model id: {} training log ===\n'.format(model_id)
        score_line = 'score: {:.3f}\n'.format(100*score)

        fpath = os.path.join(self.saved_models_dir, '{}_Training_Logs.txt'.format(self.model_name))
        with open(fpath, 'a') as f:
            f.writelines([top_line, score_line, bottom_line])



    def trainModel(self, model_id=1, save_model=True, **hyperparams):

        # hyperparameters
        # ================================================================================================
        cfg = self.cfg.clone()

        # set hyperparameters
        if cfg.verbose and len(hyperparams):
            print('Hyperparameters:')
        for key, val in hyperparams.items():
            subkeys = key.split('/')
            hyp = cfg
            for subkey in subkeys[:-1]:
                hyp = hyp.get(subkey)
            hyp.__setattr__(subkeys[-1], val)
            if self.cfg.verbose:
                print('{}: {}'.format(key, hyp.get(subkeys[-1])))
        '''if verbose and len(hyperparams):
            pprint.pprint(config_params)'''

        # model id
        # ================================================================================================
        model_identifier = self.model_name + '-{}'.format(model_id)

        if self.cfg.verbose:
            print('Building Model: {}'.format(model_identifier))

        # ================================================================================================

        # dataset
        # ================================================================================================
        dataset = getattr(datasets, cfg.dataset.name)(dataset_dir=cfg.dataset.root_path,
                                                           verbose=cfg.verbose)

        preprocessing = getattr(utilities.dataset_utils,
                                cfg.dataset.preprocessing.method)(
            mean_image=dataset.mean,
            std_image=dataset.std,
            **cfg.model.backbone.get(cfg.model.backbone.arch).input_parameters,
            **cfg.dataset.preprocessing.get(cfg.dataset.preprocessing.method)
        )

        representative_lifetime = 0
        sampling = datasets.MPerClass(classes_per_batch=cfg.training.classes_per_batch,
                                      sample_per_class=cfg.training.sample_per_class,
                                      representative_lifetime=representative_lifetime)

        val_subset = 'val' if dataset.size.val > 0 else 'eval'
        val_subset = 'trainval' if cfg.model.num_models > 1 else val_subset
        print(dataset.information('Using {} set to monitor training...'.format(val_subset)))

        train_data = dataset.makeBatch(subset='train',
                                       split_id=model_id,
                                       num_splits=cfg.model.num_models,
                                       sampling_fn=sampling,
                                       preprocess_fn=preprocessing)

        val_data = dataset.makeBatch(subset=val_subset,
                                     split_id=model_id,
                                     num_splits=cfg.model.num_models,
                                     preprocess_fn=preprocessing,
                                     batch_size=cfg.validation.batch_size)

        # model
        # ================================================================================================
        backbone_arch = cfg.model.backbone.arch
        backbone_cfg = cfg.model.backbone.get(backbone_arch)
        if backbone_cfg.arch_parameters.use_pretrained and (
                len(backbone_cfg.arch_parameters.pretrained_model_file) > 0):
            backbone_cfg.arch_parameters.pretrained_model_file += '-{}.h5'.format(model_id)
        dml_model = models.genericEmbeddingModel(cfg)
        dml_model.ensemble_id = model_id
        dml_model.num_classes = dataset.num_classes.train_split[dml_model.ensemble_id - 1]
        #model.summary(line_length=150)

        # loss and metric
        # ================================================================================================
        loss = getattr(losses, cfg.loss.function)(model=dml_model, cfg=cfg)

        metric = getattr(metrics, cfg.validation.metric)(
            **cfg.validation.get(cfg.validation.metric),
            normalize_embeddings=cfg.loss.computation_head.normalize_embeddings,
            lipschitz_cont=cfg.loss.computation_head.lipschitz_cont,
            split_at=dataset.split_eval_at
        )

        metric_callback = metrics.GlobalMetric(metrics=metric,
                                               feature_ends='EmbeddingHead',
                                               val_datasets=val_data,
                                               batch_size=cfg.validation.batch_size,
                                               verbose=cfg.verbose)
        dml_model.training_callbacks.append(metric_callback)

        # early stopping
        # ================================================================================================
        early_stopping_callback = \
            tf.keras.callbacks.EarlyStopping(monitor=metric.name,
                                             min_delta=cfg.training.min_improvement_margin,
                                             patience=cfg.training.early_stopping_patience,
                                             verbose=cfg.verbose,
                                             mode='max',
                                             restore_best_weights=True)
        dml_model.training_callbacks.append(early_stopping_callback)

        '''checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=paths['DEFAULT']['TRAINING_PATH'],
                                                                     monitor=metric.name,
                                                                     verbose=1,
                                                                     save_best_only=True,
                                                                     mode='max')'''


        # visualization
        # =================================================================================================
        logdir = os.path.join(self.model_save_dirs[model_id-1], 'logs', model_identifier)
        tensorboard_callback = \
            tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                           write_graph=False,
                                           histogram_freq=0,
                                           update_freq='epoch',
                                           profile_batch=0)
        dml_model.training_callbacks.append(tensorboard_callback)
        print('\033[3;34m' + 'INFO:Optimization:EarlyStopping: ' + 'Monitoring {} for early stopping...'.format(
            metric.name) + '\033[0m')

        # optimizer
        # =================================================================================================
        optimizer_class = getattr(Optimizers, cfg.optimizer.method)

        # gradient transformers: clip, norm, weight decay
        updateGradientTransformers(model=dml_model, **cfg.optimizer.gradient_transformers, mode='training')

        # lr schedulers
        lr_schedule = cfg.optimizer.learning_rate
        if cfg.optimizer.learning_rate_scheduler.method == 'reduce_on_plateau':
            reduce_lr_on_plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=metric.name,
                mode='max',
                verbose=1,
                **cfg.optimizer.learning_rate_scheduler.reduce_on_plateau
            )
            dml_model.training_callbacks.append(reduce_lr_on_plateau_callback)
            print('\033[3;34m' + 'INFO:Optimization:LR: ' + 'Monitoring {} to reduce on plateau...'.format(metric.name) + '\033[0m')

        if cfg.optimizer.learning_rate_scheduler.method == 'exponential_decay':
            info_text = 'Using scheduler with {} decay rate\n'.format(
                cfg.optimizer.learning_rate_scheduler.exponential_decay.decay_rate)
            print('\033[3;34m' + 'INFO:Optimization:LR: ' + info_text + '\033[0m')
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr_schedule,
                **cfg.optimizer.learning_rate_scheduler.exponential_decay)

        if cfg.optimizer.learning_rate_scheduler.method == 'constant':
            print('\033[3;34m' + 'INFO:Optimization:LR: Constant learning rate...' + '\033[0m')

        # lr multipliers
        cfg.optimizer.AdamLRM.lr_multiplier.update(dml_model.learning_rate_multipliers)

        # construct optimizer
        optimizer = optimizer_class(**cfg.optimizer.get(cfg.optimizer.method),
                                    learning_rate=lr_schedule,
                                    gradient_transformers=dml_model.gradient_transformers)

        # warm up training
        # ================================================================================================
        steps_per_epoch = cfg.training.steps_per_epoch
        if steps_per_epoch is None:
            steps_per_epoch = dataset.size.train_split[model_id - 1] // sampling.batch_size


        # warm up
        warm_start = cfg.training.warm_start
        if warm_start > 0:

            if cfg.training.freeze_during_warmup:
                excluded = cfg.training.exclude_freeze
                self.freezeModel(dml_model.arch, excluded=excluded)

            # build warm-up optimizer
            updateGradientTransformers(model=dml_model, **cfg.optimizer.gradient_transformers, mode='warm_up')
            if hasattr(lr_schedule, 'initial_learning_rate'):
                lr_schedule.initial_learning_rate *= cfg.optimizer.warm_up_learning_rate_multiplier
            else:
                lr_schedule *= cfg.optimizer.warm_up_learning_rate_multiplier

            warm_up_optimizer = optimizer_class(**cfg.optimizer.get(cfg.optimizer.method),
                                                learning_rate=lr_schedule,
                                                gradient_transformers=dml_model.warm_up_gradient_transformers)

            # compile model for training
            dml_model.arch.compile(optimizer=warm_up_optimizer,
                                   # Loss function to minimize
                                   loss=loss,
                                   # List of metrics to monitor
                                   metrics=metric)

            if warm_start < 1:
                epochs = 1
                reduced_steps_per_epoch = int(round(warm_start * steps_per_epoch))
            else:
                epochs = int(warm_start)
                reduced_steps_per_epoch = steps_per_epoch

            dml_model.arch.fit(train_data, epochs=epochs, steps_per_epoch=reduced_steps_per_epoch,
                               callbacks=dml_model.training_callbacks,
                               verbose=bool(cfg.verbose))

            # defrost if the backbone is frozen
            if cfg.training.freeze_during_warmup:
                excluded = cfg.training.exclude_defrost
                if 'freeze_bn' in cfg.model.backbone.get(cfg.model.backbone.arch).arch_parameters.keys():
                    if cfg.model.backbone.get(cfg.model.backbone.arch).arch_parameters.freeze_bn:
                        excluded.append('BatchNormalization')
                print('excluded in defrost: {}'.format(excluded))

                # unfreeze model
                self.defrostModel(dml_model.arch, excluded=excluded)

        # Compile model by specifying the training configuration (optimizer, loss, metrics)
        # ================================================================================================
        dml_model.arch.compile(optimizer=optimizer,
                               # Loss function to minimize
                               loss=loss,
                               # List of metrics to monitor
                               metrics=metric)

        # actual long term training
        # ================================================================================================
        if cfg.verbose > 1:
            dml_model.arch.summary(line_length=150) #(line_length=150, expand_nested=True)
        if os.path.exists(logdir):
            shutil.rmtree(logdir)
        file_writer = tf.summary.create_file_writer(logdir=logdir)
        file_writer.set_as_default()

        epochs = cfg.training.max_epochs

        train_history = dml_model.arch.fit(train_data, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                           callbacks=dml_model.training_callbacks,
                                           verbose=bool(cfg.verbose))

        score = max(train_history.history[metric.name])

        save_file = None
        if save_model:
            training_dir = self.model_save_dirs[model_id-1]

            save_file_prefix = os.path.join(training_dir, model_identifier + '-')
            k = 1
            save_file = save_file_prefix + '{}.h5'.format(k)
            while os.path.exists(save_file):
                k += 1
                save_file = save_file_prefix + '{}.h5'.format(k)

            emb_model = tf.keras.Model(inputs=dml_model.arch.inputs, outputs=dml_model.arch.outputs)
            emb_model.compile()
            emb_model.save(save_file, overwrite=True, include_optimizer=False, save_format='h5')

        self.logSingleTraining(score, model_id)

        return score, save_file

    def trainXValModel(self, save_model=True, verbose=0, **hyperparams):

        self.cfg.verbose = verbose


        num_models = self.cfg.model.num_models
        # ================================================================================================

        total_score = 0.0

        for k in range(num_models):

            model_score, model_file = self.trainModel(model_id=k+1,
                                                      save_model=save_model,
                                                      **hyperparams)
            total_score += model_score
            self.trained_models[k].append(model_file)

        average_score = total_score / num_models

        self.logTraining(average_score, **hyperparams)
        return -average_score # - for hyperparameter optimization

    # wrapper for hyperparameter optimization
    def singleModelFitnessScore(self, model_id=1, verbose=0, **hyperparams):
        self.cfg.verbose = verbose
        model_score, model_name = self.trainModel(model_id=model_id,
                                                  save_model=False,
                                                  **hyperparams)

        self.logTraining(model_name, model_score, **hyperparams)
        score = - model_score # since minimization
        return score

    def evaluateModel(self, verbose=0, pca_dim=0, default_dataset=True):

        dataset_id = self.cfg.dataset.name
        if default_dataset:
            dataset_id = self.model_name.split('Loss_')[-1].split('_Dataset')[0]

        metric_id = self.cfg.validation.metric

        if verbose:
            print('evaluating {}'.format(self.model_name))

        # dataset
        # ================================================================================================
        dataset = getattr(datasets, dataset_id)(dataset_dir=self.cfg.dataset.root_path,
                                                verbose=self.cfg.verbose)

        preprocessing = getattr(utilities.dataset_utils,
                                self.cfg.dataset.preprocessing.method)(
            mean_image=dataset.mean,
            std_image=dataset.std,
            **self.cfg.model.backbone.get(self.cfg.model.backbone.arch).input_parameters,
            **self.cfg.dataset.preprocessing.get(self.cfg.dataset.preprocessing.method)
        )

        val_data = dataset.makeBatch(subset='eval',
                                     preprocess_fn=preprocessing,
                                     batch_size=self.cfg.validation.batch_size)

        metric = getattr(metrics, metric_id)(
            **self.cfg.validation.get(self.cfg.validation.metric),
            normalize_embeddings=self.cfg.loss.computation_head.normalize_embeddings,
            lipschitz_cont=self.cfg.loss.computation_head.lipschitz_cont,
            split_at=dataset.split_eval_at
        )

        if verbose:
            print('computing labels')

        s = 0
        dataset_size = 0
        lbls = []
        for _, lbl in val_data:
            dataset_size += lbl.shape[0]
            lbls.append(tf.convert_to_tensor(lbl))
            s += 1

        steps = s
        labels = tf.expand_dims(tf.concat(lbls, axis=0), axis=1)

        if verbose:
            print('building metric computing graph')

        metric.buildGlobalComputeGraph(dataset_size, 128)
        to_tensor = tf.keras.layers.Lambda(lambda x: x)

        tfm_fn = lambda x: x
        if pca_dim > 0:
            pca = PCA(n_components=pca_dim, svd_solver='full')
            tfm_fn = lambda x: pca.fit_transform(x)

        # evaluating models
        features = []
        num_models = len(self.trained_models)
        k = 1
        for models in self.trained_models:
            l = 1
            num_repeat = len(models)
            model_features = []
            for model_file in models:

                if verbose:
                    print(model_file)
                    print('loading model {}/{}-{}/{}...'.format(l, num_repeat, k, num_models))
                model = tf.keras.models.load_model(model_file, compile=False)
                if verbose:
                    print('computing embedding vectors...')
                emb = model.predict(val_data, verbose=int(bool(verbose)), steps=steps)

                emb = tfm_fn(emb)

                if self.cfg.loss.computation_head.normalize_embeddings:
                    if self.cfg.loss.computation_head.lipschitz_cont:
                        emb = layers.L2Normalization()(emb)
                    else:
                        emb = tf.nn.l2_normalize(emb, axis=-1)

                model_features.append(emb)

                del model

                l += 1

            features.append(model_features)

            k += 1

        if verbose:
            print('computing results...')


        features = [f for f in features if len(f) > 0]
        num_models = len(features)
        num_repeats = [len(f) for f in features]

        per_model_metrics = []
        k = 1
        for feat_set in features:
            computed_metrics = []
            monitored_metrics = []
            l = 1
            for feats in feat_set:
                if verbose:
                    print('per model metric computing {}/{}-{}/{}...'.format(l, num_repeats[k-1], k, num_models))
                with tf.device('/device:CPU:0'):
                    computed_metric = metric.computeGlobal(to_tensor(feats),
                                                           labels)

                computed_metrics.append(computed_metric)

                if verbose > 1:
                    metric.printResult(computed_metric)

                l += 1

            per_model_metrics.append(computed_metrics)

            k += 1


        # ensemble
        grids = np.meshgrid(*tuple([np.arange(n) for n in num_repeats]), indexing='ij')
        indexing = np.stack([g.flatten() for g in grids], axis=1)

        concatenated_metrics = []
        separated_metrics = []

        for i, index in enumerate(indexing):
            if verbose:
                print('ensemble metric computing {}/{}...'.format(i+1, indexing.shape[0]))
            concatenated_feature = tf.concat([features[k][l] for k, l in enumerate(index)], axis=-1)
            if self.cfg.loss.computation_head.normalize_embeddings:
                if self.cfg.loss.computation_head.lipschitz_cont:
                    concatenated_feature = layers.L2Normalization()(concatenated_feature)
                else:
                    concatenated_feature = tf.nn.l2_normalize(concatenated_feature, axis=-1)

            with tf.device('/device:CPU:0'):
                computed_metric = metric.computeGlobal(concatenated_feature,
                                                       labels)

            concatenated_metrics.append(computed_metric)

            separated_metrics.append(
                np.mean(
                    tf.stack([per_model_metrics[k][l] for k, l in enumerate(index)], axis=0).numpy(),
                    axis=0)
            )

        concatenated_metrics = tf.stack(concatenated_metrics, axis=0).numpy()
        separated_metrics = tf.stack(separated_metrics, axis=0).numpy()

        if self.cfg.validation.get(self.cfg.validation.metric).monitored_metric.lower() == 'recall':
            monitor_id = 2
        elif self.cfg.validation.get(self.cfg.validation.metric).monitored_metric.lower() == 'precision':
            monitor_id = 1
        elif self.cfg.validation.get(self.cfg.validation.metric).monitored_metric.lower() == 'map':
            monitor_id = 0
        else:
            monitor_id = 0

        best_id = np.argmax(separated_metrics, axis=0)[monitor_id]
        worst_id = np.argmin(separated_metrics, axis=0)[monitor_id]

        best_concatenated = concatenated_metrics[best_id]
        best_separated = separated_metrics[best_id]

        worst_concatenated = concatenated_metrics[worst_id]
        worst_separated = separated_metrics[worst_id]

        avg_concatenated = np.mean(concatenated_metrics, axis=0)
        std_concatenated = np.std(concatenated_metrics, axis=0)

        avg_separated = np.mean(separated_metrics, axis=0)
        std_separated = np.std(separated_metrics, axis=0)


        print('===================================')
        print('worst:')
        print('concatenated:')
        wc_text = metric.printResult(worst_concatenated)
        print('===================================')
        print('separate:')
        ws_text = metric.printResult(worst_separated)
        print('===================================')
        print('===================================')

        print('best:')
        print('concatenated:')
        bc_text = metric.printResult(best_concatenated)
        print('===================================')
        print('separate:')
        bs_text = metric.printResult(best_separated)
        print('===================================')
        print('===================================')

        print('average:')
        print('concatenated:')
        ac_text = metric.printResult(avg_concatenated)
        sc_text = metric.printResult(std_concatenated)
        print('===================================')
        print('separate:')
        as_test = metric.printResult(avg_separated)
        ss_text = metric.printResult(std_separated)

        line_text = '==================================='
        out_text = line_text + '\n' +  'worst: \n' + \
                   'concatenated:\n{}\n{}\n'.format(wc_text, line_text) + \
                   'separate:\n{}\n{}\n'.format(ws_text, line_text) + \
                   line_text + '\n' + 'best: \n' + \
                   'concatenated:\n{}\n{}\n'.format(bc_text, line_text) + \
                   'separate:\n{}\n{}\n'.format(bs_text, line_text) + \
                   line_text + '\n' + 'average: \n' + \
                   'concatenated:\n{}\n{}\n{}\n'.format(ac_text, sc_text, line_text) + \
                   'separate:\n{}\n{}\n{}\n'.format(as_test, ss_text, line_text)

        fpath = os.path.join(self.saved_models_dir, '{}_Evaluation.txt'.format(self.model_name))
        with open(fpath, 'w') as f:
            f.write(out_text)

        return separated_metrics, concatenated_metrics


import tensorflow as tf

from sklearn.decomposition import PCA

import os
import shutil
import pprint

from framework import datasets
from framework import losses
from framework import metrics
from framework import models
from framework import layers
from framework import utilities
from framework import configs
from framework import solvers

def addRegularizationLoss(model, loss_fn):

    for layer in model.layers:

        if hasattr(layer, 'layers'):
            addRegularizationLoss(layer, loss_fn)

        if hasattr(layer, 'kernel'):
            model.add_loss(lambda layer=layer: loss_fn(layer.kernel))

def freezeModel(model, excluded=['feature_transform']):

    for layer in model.layers:
        if hasattr(layer, 'layers'):
            freezeModel(layer, excluded)

        elif not any([((e in str(layer.__class__)) or (e in str(layer.name))) for e in excluded]):
            layer.trainable = False

def unfreezeModel(model, excluded=['BatchNormalization']):

    for layer in model.layers:

        if hasattr(layer, 'layers'):
            unfreezeModel(layer, excluded)

        elif not any([((e in str(layer.__class__)) or (e in str(layer.name))) for e in excluded]):
            layer.trainable = True

def logTraining(model_name, score, training_path, params, **hyperparams):

    for key, val in hyperparams.items():
        subkeys = key.split('/')
        hyp = params
        for subkey in subkeys[:-1]:
            hyp = hyp[subkey]
        hyp[subkeys[-1]] = val

    loss_id = params['loss']['function']
    optimizer_id = params['optimizer']['method']

    top_line = '\n=== training configuration ===\n'
    bottom_line = '=== end of training configuration ===\n'
    training_line = 'optimizer: {}, learning_rate: {}, weight_decay: {}\n'.format(
        optimizer_id,
        params['optimizer']['learning_rate'],
        params['optimizer']['gradient_transformers']['weight_decay'])

    loss_common_line = 'losses:common: {}\n'.format(params['loss']['computation_head'])
    loss_line = 'losses:{}: {}\n'.format(loss_id, params['loss']['parameters'])
    profs_line = 'PROFS: {}\n'.format(params['profs'])
    score_line = 'score: {}\n'.format(score)

    fpath = os.path.join(training_path, '{}.txt'.format(model_name))
    with open(fpath, 'a') as f:
        f.writelines([top_line, training_line, loss_common_line, loss_line, profs_line, score_line, bottom_line])

def trainModel(paths, config_params, model_id=1, save_model=True, verbose=0, **hyperparams):

    params = config_params

    # set hyperparameters
    if verbose and len(hyperparams):
        print('Hyperparameters:')
    for key, val in hyperparams.items():
        subkeys = key.split('/')
        hyp = params
        for subkey in subkeys[:-1]:
            hyp = hyp[subkey]
        hyp[subkeys[-1]] = val
        if verbose:
            print('{}: {}'.format(key, hyp[subkeys[-1]]))
    '''if verbose and len(hyperparams):
        pprint.pprint(config_params)'''


    '''if verbose:
        for key in params.keys():
            print(params[key])'''


    dataset_id = params['dataset']
    loss_id = params['loss']['function']
    optimizer_id = params['optimizer']['method']
    metric_id = params['validation']['metric']

    backbone = params['model']['structure']['backbone']
    num_models = params['model']['structure']['num_models']
    # ================================================================================================
    proxy_text = 'Proxy_' if (bool(params['loss']['computation_head']['use_proxy']) or
                              (bool(params['profs']['use_profs_in_training']) and
                               bool(params['profs']['use_representative_proxy']))) else ''
    profs_text = 'PROFS_' if bool(params['profs']['use_profs_in_training']) else ''
    xbm_text = 'XBM_' if bool(params['xbm']['use_xbm_in_training']) else ''
    pooling_text = 'TransportPool' \
        if params['model']['feature_pooling']['use_transport_layer'] else 'AvgPool'
    ensemble_text = 'Ensemble_{}'.format(num_models) if num_models > 1 else 'Single'
    batch_text = '{}x{}'.format(params['training']['classes_per_batch'],
                                params['training']['sample_per_class'])

    model_identifier = backbone + '_' \
                       + '{}'.format(params['model']['embedding_head']['embedding_size']) + '_' \
                       + proxy_text + profs_text + xbm_text \
                       + loss_id + '_' \
                       + pooling_text + '_' \
                       + dataset_id + '_' \
                       + batch_text + '_' \
                       + ensemble_text + '_' \
                       + 'Model-{}'.format(model_id)

    if verbose:
        print('Training Model: {}'.format(model_identifier))

    # ================================================================================================

    # dataset
    # ================================================================================================
    dataset = getattr(datasets, dataset_id)(dataset_dir=paths['DEFAULT']['DATASETS_PATH'], verbose=verbose)


    '''value_scaler = configs.default.backbone_models[backbone]['image_scaler']
    reverse_channels = configs.default.backbone_models[backbone]['reverse_channels']'''

    preprocessing = getattr(utilities.dataset_utils,
                            params['training']['preprocessing'])(
        mean_image=dataset.mean,
        std_image=None,
        **configs.default.backbone_models[backbone]['input_parameters'])

    representative_lifetime = \
        0 if params['profs']['use_representative_proxy'] else params['profs']['profs_iterations']
    sampling = datasets.MPerClass(classes_per_batch=params['training']['classes_per_batch'],
                                  sample_per_class=params['training']['sample_per_class'],
                                  representative_lifetime=representative_lifetime,
                                  preprocess_representative=params['profs']['preprocess_representative'])

    val_subset = 'trainval' if num_models > 1 else 'eval'

    #print('\tpreparing dataset...')
    train_data = dataset.makeBatch(subset='train',
                                   split_id=model_id,
                                   num_splits=num_models,
                                   sampling_fn=sampling,
                                   preprocess_fn=preprocessing)

    val_data = dataset.makeBatch(subset=val_subset,
                                 split_id=model_id,
                                 num_splits=num_models,
                                 preprocess_fn=preprocessing,
                                 batch_size=params['validation']['batch_size'])

    #print('\tdataset is ready!')
    #print('\tbuilding model...')

    # model
    # ================================================================================================
    pretrained_model_filepath = paths['DEFAULT']['MODELS_PATH']

    model = \
        getattr(models, backbone)(model_filepath=pretrained_model_filepath,
                                  input_size=list(preprocessing.out_image_size),
                                  use_pretrained=params['model']['structure']['use_pretrained'],
                                  **configs.default.backbone_models[backbone]['model_parameters'],
                                  **params['model']['embedding_head'],
                                  **params['model']['feature_pooling'])



    #model.summary(line_length=150)

    # loss and metric
    # ================================================================================================
    loss_class = getattr(losses, loss_id)
    loss_params = \
        {'classes_per_batch': params['training']['classes_per_batch'],
         'use_ref_samples': params['profs']['use_profs_in_training'],
         'num_classes': dataset.num_classes.train_split[model_id - 1]}
    loss_params.update(params['loss']['computation_head'])
    loss_params.update(params['loss']['parameters'])

    '''loss = loss_class(model=model,
                      classes_per_batch=params['training']['classes_per_batch'],
                      use_ref_samples=params['profs']['use_profs_in_training'],
                      **params['loss']['computation_head'],
                      **params['loss']['parameters'],
                      num_classes=dataset.num_classes.train_split[model_id - 1])'''

    metric = getattr(metrics, metric_id)(
        **configs.default.metrics[metric_id],
        normalize_embeddings=params['loss']['computation_head']['normalize_embeddings'],
        lipschitz_cont=params['loss']['computation_head']['lipschitz_cont'],
        split_at=dataset.split_eval_at
    )

    metric_callback = metrics.GlobalMetric(metrics=metric,
                                           feature_ends='embedding',
                                           val_datasets=val_data,
                                           batch_size=params['validation']['batch_size'],
                                           verbose=verbose)

    # PROFS based optimization
    # ================================================================================================
    gradient_transformers = []
    model_callbacks = []
    if params['profs']['use_profs_in_training']:
        gradient_transformers, model_callbacks, loss = solvers.asap.useASAP(
            model=model,
            loss_class=loss_class,
            loss_params=loss_params,
            dataset=dataset,
            batch_size=params['profs']['proxy_sampling_batch_size'],
            subset='train',
            split_id=model_id,
            num_splits=num_models,
            preprocessing=preprocessing,
            **params['profs'])
    else:
        loss = loss_class(model=model, **loss_params)



    # callbacks
    # ================================================================================================
    early_stopping_callback = \
        tf.keras.callbacks.EarlyStopping(monitor=metric.name,
                                         min_delta=params['training']['min_improvement_margin'],
                                         patience=params['training']['early_stopping_patience'],
                                         verbose=verbose,
                                         mode='max',
                                         restore_best_weights=True)

    '''checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=paths['DEFAULT']['TRAINING_PATH'],
                                                                 monitor=metric.name,
                                                                 verbose=1,
                                                                 save_best_only=True,
                                                                 mode='max')'''

    logdir = os.path.join(paths['DEFAULT']['TRAINING_PATH'], 'logs', model_identifier)
    tensorboard_callback = \
        tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                       write_graph=False,
                                       histogram_freq=0,
                                       update_freq='epoch',
                                       profile_batch=0)

    # lr scheduler callback
    if params['optimizer']['learning_rate_scheduler']['decay_rate'] < 1.0:
        reduce_lr_on_plateau_callback = []
    else:
        reduce_lr_on_plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=metric.name,
            mode='max',
            verbose=1,
            **params['optimizer']['reduce_lr_on_plateau']
        )
        print('\nLR: monitor {} to reduce on plateau\n'.format(metric.name))

    # optimizer
    # ================================================================================================
    gradient_transformers += solvers.gradient_transformers.getGradientTransformers(
        model=model,
        **params['optimizer']['gradient_transformers'])
    optimizer_class = getattr(solvers.Optimizers, optimizer_id)

    if params['optimizer']['learning_rate_scheduler']['decay_rate'] < 1.0:
        print('\nLR: using scheduler with {} decay rate\n'.format(params['optimizer']['learning_rate_scheduler']['decay_rate']))
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=params['optimizer']['learning_rate'],
            **params['optimizer']['learning_rate_scheduler'])
    else:
        lr_schedule = params['optimizer']['learning_rate']

    if tf.__version__ >= '2.4':
        optimizer = optimizer_class(**params['optimizer']['parameters'],
                                    learning_rate=lr_schedule,
                                    gradient_transformers=gradient_transformers)
    else:
        optimizer = solvers.buildOptimizer(optimizer_class,
                                           **params['optimizer']['parameters'],
                                           learning_rate=lr_schedule,
                                           gradient_transformers=gradient_transformers)


    # warm up training
    # ================================================================================================
    steps_per_epoch = params['training']['steps_per_epoch']
    if steps_per_epoch is None:
        steps_per_epoch = dataset.size.train_split[model_id - 1] // sampling.batch_size

    if params['profs']['use_profs_in_training'] and not params['profs']['early_stopping']:
        steps_per_epoch -= steps_per_epoch % params['profs']['profs_iterations']

    # warm up
    warm_start = params['training']['warm_start']
    if warm_start > 0:

        '''for w in model.trainable_weights:
            print(w.name)'''

        if params['training']['freeze_during_warmup']:
            freezeModel(model, excluded=['feature_transform'])

        # Specify the training configuration (optimizer, loss, metrics)
        if tf.__version__ >= '2.4':
            warm_up_optimizer = optimizer_class(
                **params['optimizer']['parameters'],
                learning_rate=params['optimizer']['learning_rate'],
                gradient_transformers=solvers.gradient_transformers.getGradientTransformers(
                    model=model,
                    **params['optimizer']['gradient_transformers']))
        else:
            warm_up_optimizer = \
                solvers.buildOptimizer(
                    optimizer_class,
                    **params['optimizer']['parameters'],
                    learning_rate=params['optimizer']['learning_rate'],
                    gradient_transformers=solvers.gradient_transformers.getGradientTransformers(
                        model=model,
                        **params['optimizer']['gradient_transformers']))

        model.compile(optimizer=warm_up_optimizer,
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

        model.fit(train_data, epochs=epochs, steps_per_epoch=reduced_steps_per_epoch,
                  callbacks=[metric_callback, early_stopping_callback, tensorboard_callback] + model_callbacks,
                  verbose=bool(verbose))

        # unfreeze if the backbone is frozen
        if params['training']['freeze_during_warmup']:
            excluded = []
            if 'freeze_bn' in configs.default.backbone_models[backbone]['model_parameters'].keys():

                excluded = ['BatchNormalization'] \
                    if configs.default.backbone_models[backbone]['model_parameters']['freeze_bn'] else excluded
                print('excluded in defrost: {}'.format(excluded))

            # unfreeze model
            unfreezeModel(model, excluded=excluded)

    # Compile model by specifying the training configuration (optimizer, loss, metrics)
    # ================================================================================================
    model.compile(optimizer=optimizer,
                  # Loss function to minimize
                  loss=loss,
                  # List of metrics to monitor
                  metrics=metric)

    # actual long term training
    # ================================================================================================
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    file_writer = tf.summary.create_file_writer(logdir=logdir)
    file_writer.set_as_default()

    epochs = params['training']['max_epochs']

    train_history = model.fit(train_data, epochs=epochs, steps_per_epoch=steps_per_epoch,
                              callbacks=[metric_callback, early_stopping_callback, tensorboard_callback] + [reduce_lr_on_plateau_callback] + model_callbacks,
                              verbose=bool(verbose))

    score = max(train_history.history[metric.name])

    if save_model:
        training_dir = paths['DEFAULT']['TRAINING_PATH']
        save_file = os.path.join(training_dir, model_identifier + '.h5')
        model.save(save_file, overwrite=True, include_optimizer=False, save_format='h5')

    del model
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    return score, model_identifier

def trainXValModel(paths, config_params, save_model=True, verbose=0, **hyperparams):

    num_models = config_params['model']['structure']['num_models']
    # ================================================================================================

    total_score = 0.0
    model_names = []
    for k in range(num_models):

        model_score, model_name = trainModel(paths=paths,
                                             config_params=config_params,
                                             model_id=k+1,
                                             save_model=save_model,
                                             verbose=verbose, **hyperparams)
        total_score += model_score
        model_names.append(model_name)

    average_score = total_score / num_models

    model_name = model_names[0].split('-')[0]
    logTraining(model_name, average_score, paths['DEFAULT']['TRAINING_PATH'], config_params, **hyperparams)
    return average_score, model_names

# wrapper for hyperparameter optimization
def modelFitnessScore(paths, config_params, verbose=0, **hyperparams):
    score, _ = trainXValModel(paths, config_params, save_model=False, verbose=verbose, **hyperparams)
    return - score # since minimization

# wrapper for hyperparameter optimization
def singleModelFitnessScore(paths, config_params, model_id=1, verbose=0, **hyperparams):

    model_score, model_name = trainModel(paths=paths,
                                         config_params=config_params,
                                         model_id=model_id,
                                         save_model=False,
                                         verbose=verbose,
                                         **hyperparams)

    logTraining(model_name, model_score, paths['DEFAULT']['TRAINING_PATH'], config_params, **hyperparams)
    score = - model_score # since minimization
    return score

def trainFancyModel(paths, config_params, model_id=1, save_model=True, verbose=0, **hyperparams):

    params = config_params

    # set hyperparameters
    if verbose and len(hyperparams):
        print('Hyperparameters:')
    for key, val in hyperparams.items():
        subkeys = key.split('/')
        hyp = params
        for subkey in subkeys[:-1]:
            hyp = hyp[subkey]
        hyp[subkeys[-1]] = val
        if verbose:
            print('{}: {}'.format(key, hyp[subkeys[-1]]))
    '''if verbose and len(hyperparams):
        pprint.pprint(config_params)'''


    '''if verbose:
        for key in params.keys():
            print(params[key])'''


    dataset_id = params['dataset']
    loss_id = params['loss']['function']
    optimizer_id = params['optimizer']['method']
    metric_id = params['validation']['metric']

    backbone = params['model']['structure']['backbone']
    num_models = params['model']['structure']['num_models']
    # ================================================================================================
    proxy_text = 'Proxy_' if (bool(params['loss']['computation_head']['use_proxy']) or
                              (bool(params['profs']['use_profs_in_training']) and
                               bool(params['profs']['use_representative_proxy']))) else ''
    profs_text = 'PROFS_' if bool(params['profs']['use_profs_in_training']) else ''
    xbm_text = 'XBM_' if bool(params['xbm']['use_xbm_in_training']) else ''
    pooling_text = 'TransportPool' \
        if params['model']['feature_pooling']['use_transport_layer'] else 'AvgPool'
    ensemble_text = 'Ensemble_{}'.format(num_models) if num_models > 1 else 'Single'
    batch_text = '{}x{}'.format(params['training']['classes_per_batch'],
                                params['training']['sample_per_class'])

    model_identifier = backbone + '_' \
                       + '{}'.format(params['model']['embedding_head']['embedding_size']) + '_' \
                       + proxy_text + profs_text + xbm_text \
                       + loss_id + '_' \
                       + pooling_text + '_' \
                       + dataset_id + '_' \
                       + batch_text + '_' \
                       + ensemble_text + '_' \
                       + 'Model-{}'.format(model_id)

    if verbose:
        print('Training Model: {}'.format(model_identifier))

    # ================================================================================================

    # dataset
    # ================================================================================================
    dataset = getattr(datasets, dataset_id)(dataset_dir=paths['DEFAULT']['DATASETS_PATH'], verbose=verbose)


    '''value_scaler = configs.default.backbone_models[backbone]['image_scaler']
    reverse_channels = configs.default.backbone_models[backbone]['reverse_channels']'''

    preprocessing = getattr(utilities.dataset_utils,
                            params['training']['preprocessing'])(
        mean_image=dataset.mean,
        std_image=None,
        **configs.default.backbone_models[backbone]['input_parameters'])

    representative_lifetime = \
        0 if params['profs']['use_representative_proxy'] else params['profs']['profs_iterations']
    sampling = datasets.MPerClass(classes_per_batch=params['training']['classes_per_batch'],
                                  sample_per_class=params['training']['sample_per_class'],
                                  representative_lifetime=representative_lifetime,
                                  preprocess_representative=params['profs']['preprocess_representative'])

    val_subset = 'trainval' if num_models > 1 else 'eval'

    #print('\tpreparing dataset...')
    train_data = dataset.makeBatch(subset='train',
                                   split_id=model_id,
                                   num_splits=num_models,
                                   sampling_fn=sampling,
                                   preprocess_fn=preprocessing)

    val_data = dataset.makeBatch(subset=val_subset,
                                 split_id=model_id,
                                 num_splits=num_models,
                                 preprocess_fn=preprocessing,
                                 batch_size=params['validation']['batch_size'])

    #print('\tdataset is ready!')
    #print('\tbuilding model...')

    # model
    # ================================================================================================
    pretrained_model_filepath = paths['DEFAULT']['MODELS_PATH']

    model = \
        getattr(models, backbone)(model_filepath=pretrained_model_filepath,
                                  input_size=list(preprocessing.out_image_size),
                                  **configs.default.backbone_models[backbone]['model_parameters'],
                                  **params['model']['embedding_head'],
                                  **params['model']['feature_pooling'])



    #model.summary(line_length=150)

    # loss and metric
    # ================================================================================================
    loss_class = getattr(losses, loss_id)
    loss_params = \
        {'classes_per_batch': params['training']['classes_per_batch'],
         'use_ref_samples': params['profs']['use_profs_in_training'],
         'num_classes': dataset.num_classes.train_split[model_id - 1]}
    loss_params.update(params['loss']['computation_head'])
    loss_params.update(params['loss']['parameters'])

    '''loss = loss_class(model=model,
                      classes_per_batch=params['training']['classes_per_batch'],
                      use_ref_samples=params['profs']['use_profs_in_training'],
                      **params['loss']['computation_head'],
                      **params['loss']['parameters'],
                      num_classes=dataset.num_classes.train_split[model_id - 1])'''

    metric = getattr(metrics, metric_id)(
        **configs.default.metrics[metric_id],
        normalize_embeddings=params['loss']['computation_head']['normalize_embeddings'],
        lipschitz_cont=params['loss']['computation_head']['lipschitz_cont'],
        split_at=dataset.split_eval_at
    )

    metric_callback = metrics.GlobalMetric(metrics=metric,
                                           feature_ends='embedding',
                                           val_datasets=val_data,
                                           batch_size=params['validation']['batch_size'],
                                           verbose=verbose)

    # PROFS based optimization
    # ================================================================================================
    gradient_transformers = []
    model_callbacks = []
    if params['profs']['use_profs_in_training']:
        gradient_transformers, model_callbacks, loss = solvers.asap.usePROFS(
            model=model,
            loss_class=loss_class,
            loss_params=loss_params,
            dataset=dataset,
            batch_size=params['profs']['proxy_sampling_batch_size'],
            subset='train',
            split_id=model_id,
            num_splits=num_models,
            preprocessing=preprocessing,
            **params['profs'])
    else:
        loss = loss_class(model=model, **loss_params)



    # callbacks
    # ================================================================================================
    early_stopping_callback = \
        tf.keras.callbacks.EarlyStopping(monitor=metric.name,
                                         min_delta=params['training']['min_improvement_margin'],
                                         patience=params['training']['early_stopping_patience'],
                                         verbose=verbose,
                                         mode='max',
                                         restore_best_weights=True)

    '''checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=paths['DEFAULT']['TRAINING_PATH'],
                                                                 monitor='val_{}'.format(metric.name),
                                                                 verbose=1,
                                                                 save_best_only=True,
                                                                 mode='max')'''

    logdir = os.path.join(paths['DEFAULT']['TRAINING_PATH'], 'logs', model_identifier)
    tensorboard_callback = \
        tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                       write_graph=False,
                                       histogram_freq=0,
                                       update_freq='epoch',
                                       profile_batch=0)

    '''embedding_space_callback = \
        utilities.visualization_utils.ValualizationCallback(
            asap_step=model_callbacks[0].global_step,
            val_dataset=val_data,
            log_dir=os.path.join(logdir, 'embedding_space')
        )'''


    # optimizer
    # ================================================================================================
    gradient_transformers += solvers.gradient_transformers.getGradientTransformers(
        model=model,
        **params['optimizer']['gradient_transformers'])
    optimizer_class = getattr(solvers.Optimizers, optimizer_id)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=params['optimizer']['learning_rate'],
        **params['optimizer']['learning_rate_scheduler'])

    if tf.__version__ >= '2.4':
        optimizer = optimizer_class(**params['optimizer']['parameters'],
                                    learning_rate=lr_schedule,
                                    gradient_transformers=gradient_transformers)
    else:
        optimizer = solvers.buildOptimizer(optimizer_class,
                                           **params['optimizer']['parameters'],
                                           learning_rate=lr_schedule,
                                           gradient_transformers=gradient_transformers)


    # warm up training
    # ================================================================================================
    steps_per_epoch = params['training']['steps_per_epoch']
    if steps_per_epoch is None:
        steps_per_epoch = dataset.size.train_split[model_id - 1] // sampling.batch_size


    if params['profs']['use_profs_in_training'] and not params['profs']['early_stopping']:
        steps_per_epoch -= steps_per_epoch % params['profs']['profs_iterations']

    # warm up
    warm_start = params['training']['warm_start']
    if warm_start > 0:

        '''for w in model.trainable_weights:
            print(w.name)'''

        if params['training']['freeze_during_warmup']:
            freezeModel(model, excluded=['feature_transform'])

        # Specify the training configuration (optimizer, loss, metrics)
        if tf.__version__ >= '2.4':
            warm_up_optimizer = optimizer_class(
                **params['optimizer']['parameters'],
                learning_rate=params['optimizer']['learning_rate'],
                gradient_transformers=solvers.gradient_transformers.getGradientTransformers(
                    model=model,
                    **params['optimizer']['gradient_transformers']))
        else:
            warm_up_optimizer = \
                solvers.buildOptimizer(
                    optimizer_class,
                    **params['optimizer']['parameters'],
                    learning_rate=params['optimizer']['learning_rate'],
                    gradient_transformers=solvers.gradient_transformers.getGradientTransformers(
                        model=model,
                        **params['optimizer']['gradient_transformers']))


        model.compile(optimizer=warm_up_optimizer,
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

        model.fit(train_data, epochs=epochs, steps_per_epoch=reduced_steps_per_epoch,
                  callbacks=[metric_callback, early_stopping_callback, tensorboard_callback] + model_callbacks,
                  verbose=bool(verbose))

        # unfreeze if the backbone is frozen
        if params['training']['freeze_during_warmup']:
            excluded = []
            if 'freeze_bn' in configs.default.backbone_models[backbone].keys():

                excluded = ['BatchNormalization'] \
                    if configs.default.backbone_models[backbone]['freeze_bn'] else excluded
                print('excluded {}'.format(excluded))

            # unfreeze model
            unfreezeModel(model, excluded=excluded)

    # XBM
    # ================================================================================================
    if params['xbm']['use_xbm_in_training']:
        batches_in_mem = params['xbm']['batches_in_mem']
        loss_with_xbm, xbm_init_callback = xbm.useXBM(model=model,
                                                      batch_size=sampling.batch_size,
                                                      batches_in_mem=batches_in_mem,
                                                      xbm_weight=params['xbm']['xbm_weight'],
                                                      loss_weight=params['xbm']['loss_weight'],
                                                      loss_class=loss_class,
                                                      train_data=train_data,
                                                      classes_per_batch=params['training'][
                                                          'classes_per_batch'],
                                                      use_ref_samples=params['profs'][
                                                          'use_profs_in_training'],
                                                      **params['loss']['parameters'],
                                                      **params['loss']['computation_head'],
                                                      num_classes=dataset.num_classes.train_split[
                                                          model_id - 1])
        loss = loss_with_xbm
        model_callbacks = model_callbacks + xbm_init_callback



    # Compile model by specifying the training configuration (optimizer, loss, metrics)
    model.compile(optimizer=optimizer,
                  # Loss function to minimize
                  loss=loss,
                  # List of metrics to monitor
                  metrics=metric)

    # actual long term training
    # ================================================================================================
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    file_writer = tf.summary.create_file_writer(logdir=logdir)
    file_writer.set_as_default()

    epochs = params['training']['max_epochs']
    callbacks = \
        [metric_callback, early_stopping_callback, tensorboard_callback, reduce_lr_on_plateau_callback] + model_callbacks # + [embedding_space_callback]

    train_history = model.fit(train_data, epochs=epochs, steps_per_epoch=steps_per_epoch,
                              callbacks=callbacks,
                              verbose=bool(verbose))

    score = max(train_history.history[metric.name])

    if save_model:
        training_dir = paths['DEFAULT']['TRAINING_PATH']
        save_file = os.path.join(training_dir, model_identifier + '.h5')
        model.save(save_file, overwrite=True, include_optimizer=False, save_format='h5')

    del model
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    return score, model_identifier


def evaluateModel(model_name, paths, config_params, verbose=0, pca_dim=0):

    params = config_params
    dataset_id = params['dataset']
    metric_id = params['validation']['metric']

    backbone = model_name.split('_')[0]

    # parameters config
    '''parameters_config_file = paths['DEFAULT']['CONFIG_PATH']
    with open(parameters_config_file) as f:
        params = json.load(f)'''

    training_dir = paths['DEFAULT']['TRAINING_PATH']

    # get model files
    model_files = []
    for file in os.listdir(training_dir):
        if file.startswith(model_name) and file.endswith('h5'):
            model_files.append(os.path.join(training_dir, file))
    if verbose:
        print('evaluating {}'.format(model_name))

    # dataset
    # ================================================================================================
    dataset = getattr(datasets, dataset_id)(dataset_dir=paths['DEFAULT']['DATASETS_PATH'])

    '''value_scaler = configs.default.backbone_models[backbone]['image_scaler']
    reverse_channels = configs.default.backbone_models[backbone]['reverse_channels']

    preprocessing = utilities.dataset_utils.DMLPreprocessing(mean_image=dataset.mean,
                                                             std_image=None,
                                                             value_scaler=value_scaler,
                                                             reverse_channels=reverse_channels)'''

    preprocessing = getattr(utilities.dataset_utils,
                            params['training']['preprocessing'])(
        mean_image=dataset.mean,
        std_image=None,
        **configs.default.backbone_models[backbone]['input_parameters'])

    val_data = dataset.makeBatch(subset='eval',
                                 preprocess_fn=preprocessing,
                                 batch_size=params['validation']['batch_size'])

    metric = getattr(metrics, metric_id)(**configs.default.metrics[metric_id],
                                         split_at=dataset.split_eval_at)

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

    features = []
    num_models = len(model_files)
    k = 1
    for model_file in model_files:
        print(model_file)
        if verbose:
            print('loading model {}-{}...'.format(k, num_models))
        model = tf.keras.models.load_model(model_file, compile=False)#,
                                           #custom_objects={'L2Normalization': layers.L2Normalization})
        if verbose:
            print('computing embedding vectors...')
        emb = model.predict(val_data, verbose=int(bool(verbose)), steps=steps)

        emb = tfm_fn(emb)

        if params['loss']['computation_head']['normalize_embeddings']:
            if params['loss']['computation_head']['lipschitz_cont']:
                emb = layers.L2Normalization()(emb)
            else:
                emb = tf.nn.l2_normalize(emb, axis=-1)

        features.append(emb)

        del model

        k += 1

    if verbose:
        print('computing results...')

    if num_models > 1:
        concatenated_feature = tf.concat(features, axis=1)
        if params['loss']['computation_head']['normalize_embeddings']:
            if params['loss']['computation_head']['lipschitz_cont']:
                concatenated_feature = layers.L2Normalization()(concatenated_feature)
            else:
                concatenated_feature = tf.nn.l2_normalize(concatenated_feature, axis=-1)

        with tf.device('/device:CPU:0'):
            computed_metric = metric.computeGlobal(concatenated_feature,
                                                   labels)

        monitored_metric = metric.toScalar(computed_metric)

        metric.set_weights([monitored_metric])
        print('concatenated:')
        metric.printResult(computed_metric)
        print('===================================')

    avg_computed_metric = 0.
    avg_monitored_metric = 0.

    for feature in features:
        with tf.device('/device:CPU:0'):
            computed_metric = metric.computeGlobal(to_tensor(feature),
                                                   labels)

        monitored_metric = metric.toScalar(computed_metric)

        avg_computed_metric += computed_metric
        avg_monitored_metric += monitored_metric
        if verbose > 1:
            metric.printResult(computed_metric)


    avg_computed_metric /= num_models
    avg_monitored_metric /= num_models

    print('separate:')
    metric.printResult(avg_computed_metric)

    return avg_computed_metric


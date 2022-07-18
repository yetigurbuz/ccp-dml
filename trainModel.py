import sys

import os

import tensorflow as tf

from framework.solvers import DMLFramework
from framework.configs import default


def main():

    # === parameters ===============================================
    default_cfg = default.cfg
    path_to_config_files = default_cfg.root

    device_id = 0
    model_id = 'all'
    repeats = 3
    overwrite = False
    verbose = 2
    save_model = True
    evaluate_model = True
    loss_fn = None
    conventional = False
    pooling = 'GlobalPooling'
    dataset = None

    cfg = default_cfg.clone()

    if len(sys.argv) > 1:

        for arg in sys.argv:

            ''' configuration file '''
            if arg.startswith('--cfg'):
                cfg_file = arg.split('=')[-1]
                # parameters config
                cfg_file = cfg_file if cfg_file.endswith('.yaml') else cfg_file + '.yaml'
                cfg.merge_from_file(os.path.join(path_to_config_files, cfg_file))

            ''' device '''
            if arg.startswith('--device'):
                device_id = int(arg.split('=')[-1])

            ''' model id '''
            if arg.startswith('--model'):
                model_id = int(arg.split('=')[-1])

            ''' replica id '''
            if arg.startswith('--repeat'):
                repeats = int(arg.split('=')[-1])

            ''' overwrite '''
            if arg.startswith('--overwrite'):
                overwrite = True if arg.split('=')[-1].lower() in [True, 1, '1', 'true'] else False

            ''' evaluate '''
            if arg.startswith('--evaluate'):
                evaluate_model = True if arg.split('=')[-1].lower() in [True, 1, '1', 'true'] else False

            ''' save '''
            if arg.startswith('--save'):
                save_model = True if arg.split('=')[-1].lower() in [True, 1, '1', 'true'] else False

            ''' dataset id '''
            if arg.startswith('--dataset'):
                dataset = arg.split('=')[-1]

            ''' loss id '''
            if arg.startswith('--loss'):
                loss_fn = arg.split('=')[-1].split('/')

            ''' pooling '''
            if arg.startswith('--pooling'):
                pooling = arg.split('=')[-1]

            ''' conventional '''
            if arg.startswith('--conventional'):
                conventional = True if arg.split('=')[-1].lower() in [True, 1, '1', 'true'] else False




    else:
        raise ValueError('name of the config file must be provided as --cfg=config_file ')


    # ==============================================================

    # if dataset argument exists, update the fields
    if dataset is not None:
        cfg.dataset.name = dataset

    # if loss argument exists, update the fields
    dml_loss = cfg.loss.function
    if loss_fn is not None:
        loss = loss_fn.pop(0)
        cfg.loss.function = loss
        loss_ptr = cfg.loss.get(loss)

        while len(loss_fn) > 0:
            loss = loss_fn.pop(0)
            loss_ptr.function = loss
            loss_ptr = cfg.loss.get(loss)

        dml_loss = loss


    cfg.model.embedding_head.arch = pooling

    if dml_loss in ['proxy_nca', 'proxy_anchor', 'soft_triple']:
        cfg.optimizer.method = 'AdamLRM'
        cfg.training.warm_start = 1
        cfg.model.embedding_head.MetaTransportPooling.warm_up_steps = cfg.training.steps_per_epoch
        cfg.model.embedding_head.MetaTransportPooling.entropy_regularizer_weight = 0.5
        cfg.training.classes_per_batch = cfg.training.classes_per_batch * cfg.training.sample_per_class
        cfg.training.sample_per_class = 1

        # works only for conventional case. for MLRC setting, if batch size is largen than the num of classes
        # it will raise error
        if 'CUB' in cfg.dataset.name:
            cfg.training.classes_per_batch = min(cfg.training.classes_per_batch, 100)
        if 'Cars' in cfg.dataset.name:
            cfg.training.classes_per_batch = min(cfg.training.classes_per_batch, 96)

    if 'SOP' in cfg.dataset.name:
        cfg.loss.xbm.batches_in_mem = 1400
    if 'InShop' in cfg.dataset.name:
        cfg.loss.xbm.batches_in_mem = 400

    if conventional:
        model_id = 1
        cfg.model.num_models = 1

    # device to be used in training
    use_device = '/device:GPU:{}'.format(device_id)
    print('\033[3;34m' + 'INFO:Device: ' + 'Using {}...'.format(use_device) + '\033[0m')

    with tf.device(use_device):

        trainer = DMLFramework(cfg)

        if overwrite:
            print('\033[3;31m' + 'WARNING:Trainer: ' + 'Previous models are erased!\033[0m')
            trainer.clearTrainedModels()

        for r in range(repeats):
            print('\nTRAINING {}/{}\n'.format(r+1, repeats))

            if model_id == 'all':
                trainer.trainXValModel(save_model=save_model, verbose=verbose)
            else:
                trainer.trainModel(model_id=model_id, save_model=save_model)
                trainer.trained_models = trainer.getTrainedModels(suffix='-{}'.format(model_id))

        if evaluate_model:
            trainer.evaluateModel(verbose=bool(verbose), pca_dim=0, default_dataset=True)

# =======================
#      MAIN PROGRAM
# =======================

# Main Program
if __name__ == "__main__":
    # launch main script
    main()

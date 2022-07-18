from . import default

import yaml
import os

def updateConfig(base_config: dict, ref_config: dict):

    for k, v in ref_config.items():

        if k in base_config.keys():

            if isinstance(v, dict):
                updateConfig(base_config[k], v)
            else:
                base_config[k] = v
        else:
            base_config.update({k: v})

    return base_config

def updateOptimizerConfig(base_config: dict):

    method = base_config['optimizer']['method']
    if base_config['optimizer']['parameters'] == {}:
        base_config['optimizer']['parameters'] = \
            default.optimizers[method]

    if method.lower().endswith('lrm'):
        base_config['optimizer']['parameters']['lr_multiplier'].update(base_config['optimizer'].pop('lr_multiplier'))

def updateLossConfig(base_config: dict):

    # get parameters for the dataset if available
    ds_id = base_config['dataset']
    if ds_id in base_config['loss']['parameters']:
        base_config['loss']['parameters'] = \
            base_config['loss']['parameters'][ds_id]
    elif 'default' in base_config['loss']['parameters']:
        base_config['loss']['parameters'] = \
            base_config['loss']['parameters']['default']
    else:
        pass

    # update learning rate multiplier if loss has such parameters
    lrm_keys = []
    loss_function_id = base_config['loss']['function'].lower().split('loss')[0] + '_loss'
    for k in base_config['loss']['parameters'].keys():
        if k.lower().endswith('_lrm'):
            lrm_keys.append(k)
    for k in lrm_keys:
        key = '{}/{}'.format(loss_function_id,
                             k.lower().split('_lrm')[0])
        val = base_config['loss']['parameters'].pop(k)
        base_config['optimizer']['lr_multiplier'].update({key: val})


def mergeConfigs(config_dict, configs_path=''):

    if not isinstance(config_dict, dict):
        with open(os.path.join(configs_path, config_dict), 'r') as stream:
            config_dict = yaml.safe_load(stream)


    if 'base' in config_dict.keys():
        base_config = mergeConfigs(config_dict.pop('base'), configs_path)

    else:
        base_config = default.framework


    return updateConfig(base_config, config_dict)

def getConfig(config_dict, configs_path=''):

    config = mergeConfigs(config_dict, configs_path)

    updateLossConfig(config)
    updateOptimizerConfig(config)

    return config








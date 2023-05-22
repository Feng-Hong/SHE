import yaml
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from utils.logger import custom_logger
import wandb

import torch.multiprocessing as mp
import torch.distributed as dist
import train_funs 
from utils.general import random_str, get_date, re_nest_configs

WORLD_SIZE = 1
MULTIPROCESSING_DISTRIBUTED = True
DISTRIBUTED = WORLD_SIZE > 1 or MULTIPROCESSING_DISTRIBUTED
RANK = 0


def init_seeds(seed):
    print('=====> Using fixed random seed: ' + str(seed))
    # random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # ============================================================================
    # argument parser

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None, type=str, help='Indicate the config file used for the training.')
    parser.add_argument('--seed', default=None, type=int, help='Fix the random seed for reproduction. Default is 25.')
    parser.add_argument('--output_dir', default=None, type=str, help='Output directory that saves everything.')
    parser.add_argument('--log_file', default=None, type=str, help='Logger file name.')
    # phase
    parser.add_argument('--phase', default=None, type=str, help='Phase of the program. Default is train.')
    # port
    parser.add_argument('--port', default=29500, type=int, help='Port for distributed training.')

    args = parser.parse_args()

    # load config file
    print('=====> Loading config file: ' + args.cfg)
    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    print('=====> Config file loaded')

    # ============================================================================
    # fix random seed
    if args.seed is None:
        args.seed = 25
    # ============================================================================
    # update config file

    if args.phase is not None:
        config['phase'] = args.phase
    if args.seed is not None:
        config['seed']["train"] = args.seed
    if args.log_file is not None:
        config['log_file'] = args.log_file
    if args.port is not None:
        config['port'] = args.port

    # ============================================================================
    # init logger
    if args.output_dir is None:
        args.output_dir = './exp/'+ config['dataset']['name'].split("-")[0]+'/' #./datset/
        # model
        args.output_dir = args.output_dir + config['networks']['type']
        # networks: params: m_type or base_encoder
        args.output_dir = args.output_dir + '-' + config['networks']['params']['m_type'] if 'm_type' in config['networks']['params'].keys() else args.output_dir + '-' + config['networks']['params']['base_encoder']
        if 'imb' in config['dataset'].keys():
            args.output_dir = args.output_dir + '-imb_' + str(config['dataset']['imb']) # imbalance ratio
            args.output_dir = args.output_dir + '-imb_type_' + config['dataset']['imb_type']
            args.output_dir = args.output_dir + '-loss_' + config['training_opt']['loss']
        else:
            args.output_dir = args.output_dir + 'loss_' + config['training_opt']['loss']
        args.output_dir = args.output_dir +  '-' + config['training_opt']['type']
        # optimizer
        args.output_dir = args.output_dir + '-optim_' + config['training_opt']['optimizer']
        # bs
        args.output_dir = args.output_dir +  '-bs_' + str(config['training_opt']['batch_size'])
        # epochs
        args.output_dir = args.output_dir +  '-epochs_' + str(config['training_opt']['num_epochs'])
        # lr
        args.output_dir = args.output_dir +  '-lr_' + str(config['training_opt']['optim_params']['lr'])
        args.output_dir = args.output_dir +  '-wd_' + str(config['training_opt']['optim_params']['weight_decay'])
        # sampler
        args.output_dir = args.output_dir +  '-sampler_' + config['sampler']
        # seed
        args.output_dir = args.output_dir +  '-seed_t' + str(config['seed']['train']) + '_d' + str(config['seed']['dataset'])
        # notes
        args.output_dir = args.output_dir +  '-' + config['notes'] if "notes" in config else args.output_dir
        args.output_dir = args.output_dir + '/' + get_date()+ '_' + random_str(6)
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
        # config.update({'output_dir': args.output_dir})
    os.makedirs(args.output_dir, exist_ok=True)
    # run.name = wandb_config['dataset']['name'] + '_' + args.output_dir.split('/')[-2]
    if args.log_file is None:
        logger = custom_logger(args.output_dir)
    else:
        logger = custom_logger(args.output_dir, args.log_file)

    logger.info('========================= Start Main =========================')
    # ============================================================================
    # save config file
    logger.info('=====> Saving config file')
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info('=====> Config file saved')

    # ============================================================================
    # ddp
    ngpus_per_node = torch.cuda.device_count()
    if MULTIPROCESSING_DISTRIBUTED:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config["world_size"] = ngpus_per_node * WORLD_SIZE
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, logger))
    else:
        # Simply call main_worker function
        main_worker(0, ngpus_per_node, config, logger)

    # ============================================================================
    # end
    logger.info('========================= End Main =========================')

def main_worker(gpu, ngpus_per_node, config, logger):
    config['gpu'] = gpu
    config['ngpus_per_node'] = ngpus_per_node
    config['multiprocessing_distributed'] = MULTIPROCESSING_DISTRIBUTED
    config['distributed'] = DISTRIBUTED

    init_seeds(config['seed']['train'] + gpu)
    if gpu is not None:
        logger.info("Use GPU: {} for training".format(gpu))
    
    if gpu % ngpus_per_node == 0:
        print('wandb init')
        run = wandb.init(config=config,project="group_imbalance")
        re_nest_configs(run.config)
        wandb.define_metric('acc', 'max')
        run.name = config['dataset']['name'] + '_' + config['output_dir'].split('/')[-2]

    if DISTRIBUTED:
        if MULTIPROCESSING_DISTRIBUTED:
            config['rank'] = RANK * ngpus_per_node + gpu
            # batch size
            config['training_opt']['batch_size'] = config['training_opt']['batch_size'] // config['world_size']
        dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{config["port"]}', world_size=config["world_size"], rank=config['rank'])
        
    # ============================================================================
    # train
    if config['phase'] == 'train':
        train_fun = getattr(train_funs, config['training_opt']['type'])(config, logger, eval=True)

        train_fun.run()

if __name__ == '__main__':
    main()






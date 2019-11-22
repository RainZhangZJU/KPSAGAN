import os.path as osp
import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default= 'options/test/test_KPSAGAN.yml',help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
           # util.mkdir_and_rename(
                #opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)


    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)
        if rank <= 0:
            logger.info('Number of val images in [{:s}]: {:d}'.format(
                dataset_opt['name'], len(val_set)))

    #### create model
    model = create_model(opt)


    avg_psnr = 0.0
    idx = 0
    dataset_dir = '/srv/wuyichao/Super-Resolution/KPSAGAN/BasicSR-master/BasicSR-master-c/result_600000/'
    util.mkdir(dataset_dir)
    for val_data in val_loader:
        idx += 1
        img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
        logger.info(img_name)
        #img_dir = os.path.join(opt['path']['val_images'], img_name)
        #util.mkdir(img_dir)

        model.feed_data(val_data)
        model.test()

        visuals = model.get_current_visuals()
        sr_img = util.tensor2img(visuals['SR'])  # uint8
        gt_img = util.tensor2img(visuals['GT'])  # uint8

        # save images
        suffix = 'cut'
        #opt['suffix']
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '.png')
        util.save_img(sr_img, save_img_path)

        # calculate PSNR
        crop_size = opt['scale']
        gt_img = gt_img / 255.
        sr_img = sr_img / 255.
        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
        avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)

    avg_psnr = avg_psnr / idx

    # log
    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    logger_val = logging.getLogger('val')  # validation logger
    logger_val.info('psnr: {:.4e}'.format(avg_psnr))



if __name__ == '__main__':
    main()

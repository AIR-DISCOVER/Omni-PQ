import datetime
from ARKitScenes.arkitscenes_dataset import ARKitSceneDataset

from models.dump_helper import dump_pc
from models.dump_helper_quad import dump_results_quad

import os
import sys
import numpy as np
import json
import argparse
import random

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

RUN_NAME = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

torch.autograd.set_detect_anomaly(True)
FLAGS = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))


from utils.lr_scheduler import get_scheduler
from utils.logger import setup_logger
from models.pq_transformer import PQ_Transformer
from models.loss_helper_pq import get_loss
from models.ap_helper_pq import APCalculator, parse_predictions, parse_groundtruths,QUADAPCalculator, parse_quad_predictions,parse_quad_groundtruths


def parse_option():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
    parser.add_argument('--quad_num_target', type=int, default=256, help='Quad proposal number [default: 256]')
    parser.add_argument('--sampling', default='vote', type=str, help='Query points sampling method (kps, fps)')

    # Transformer
    parser.add_argument('--nhead', default=8, type=int, help='multi-head number')
    parser.add_argument('--num_decoder_layers', default=6, type=int, help='number of decoder layers')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help='dim_feedforward')
    parser.add_argument('--transformer_dropout', default=0.1, type=float, help='transformer_dropout')
    parser.add_argument('--transformer_activation', default='relu', type=str, help='transformer_activation')

    # Data
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
    parser.add_argument('--dataset', default='scannet', help='Dataset name. [default: scannet]')
    parser.add_argument('--num_point', type=int, default=40000, help='Point Number [default: 50000]')
    parser.add_argument('--use_height', action='store_true', help='Use height signal in input.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--arkit', action="store_true", help="Whether or not to use ARKitScenes dataset.")

    # Dataset Splitting
    parser.add_argument('--start_proportion', default=0.00, type=float, help='Start proportion of the dataset')
    parser.add_argument('--end_proportion', default=1.00, type=float, help='End proportion of the dataset')

    # Training
    parser.add_argument('--start_epoch', type=int, default=1, help='Epoch to run [default: 1]')
    parser.add_argument('--max_epoch', type=int, default=600, help='Epoch to run [default: 180]')
    parser.add_argument('--optimizer', type=str, default='adamW', help='optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Optimization L2 weight decay [default: 0.0005]')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='Initial learning rate for all except decoder [default: 0.004]')
    parser.add_argument('--decoder_learning_rate', type=float, default=0.0001,
                        help='Initial learning rate for decoder [default: 0.0004]')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup-epoch', type=int, default=-1, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[900,1000], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1, help='Default bn momeuntum')
    parser.add_argument('--syncbn', action='store_true', help='whether to use sync bn')

    # Weak loss
    parser.add_argument("--gamma_mixture", action="store_true", help="Whether to enable gamma mixture loss.")
    parser.add_argument("--ema", action='store_true', help="whether to enable Mean Teacher strategy.")
    parser.add_argument('--ema_decay',  type=float,  default=0.999, metavar='ALPHA', help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency_weight', type=float, default=0.05, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency_rampup', type=int,  default=1,  metavar='EPOCHS', help='length of the consistency loss ramp-up')
    parser.add_argument('--lambda_metric_normal', type=float, default=0.5)
    parser.add_argument('--lambda_metric_vertical', type=float, default=0.5)
    parser.add_argument('--lambda_metric_size', type=float, default=0.5)
    parser.add_argument('--lambda_metric_score', type=float, default=0.05)
    parser.add_argument('--lambda_arkit_pc_loss', type=float, default=0.0000)

    # io
    parser.add_argument('--checkpoint_path', help='Model checkpoint path [default: None]')
    parser.add_argument('--log_dir', default=f'log/{RUN_NAME}', help='Dump dir to save model checkpoint [default: log]')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--val_freq', type=int, default=1, help='val frequency')
    parser.add_argument('--step_freq', type=int, default=1, help='step frequency')

    # others
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25], nargs='+',  #0.5
                        help='A list of AP IoU thresholds [default: 0.25,0.5]')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')
    parser.add_argument("--pc_loss", action='store_true', help='pc_loss')
    parser.add_argument("--dump_result", action='store_true', help='pc_loss')
    parser.add_argument("--is_eval_debug", action="store_true", help="Enter evaluation mode and embed.")
    parser.add_argument("--is_train_debug", action="store_true", help="Enter train mode and embed.")

    # Eval
    parser.add_argument("--nms_iou_quad", type=float, default=0.25, help="NMS threshold for quad.")

    args = parser.parse_args()
    # args, unparsed = parser.parse_known_args()
    args.print_freq = int(args.print_freq / args.end_proportion)
    args.save_freq = int(args.save_freq / args.end_proportion)
    args.val_freq = int(args.val_freq / args.end_proportion)
    args.max_epoch = int(args.max_epoch / args.end_proportion)
    args.consistency_rampup = int(args.consistency_rampup / args.end_proportion)

    global FLAGS
    FLAGS = {}
    FLAGS['args'] = args

    return args


def initiate_environment(args):
    '''
    initiate randomness.
    :param config:
    :return:
    '''
    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)
    np.random.seed(args.rng_seed)
    random.seed(args.rng_seed)
    

def load_checkpoint(args, model, optimizer, scheduler, **kwargs):
    logger.info("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    if checkpoint['epoch'] == 'last':
        checkpoint['epoch'] = 600
    if checkpoint['epoch'] == 'best':
        checkpoint['epoch'] = 0

    if checkpoint['epoch'] == 'ema_best':
        args.start_epoch = 1
        model.module.load_state_dict(checkpoint['ema_model'].state_dict())
    else:
        args.start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
 
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    if args.ema:
        if 'ema_model' in kwargs.keys():
            if 'ema_model' in checkpoint.keys():
                kwargs['ema_model'].load_state_dict((checkpoint['ema_model']).state_dict())
            else:
                logger.info("Loading for ema_model...")
                kwargs['ema_model'].load_state_dict({k[len("module."):]:v for k, v in checkpoint['model'].items()})

    logger.info("=> loaded successfully '{}' (epoch {})".format(args.checkpoint_path, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False, **kwargs):
    logger.info('==> Saving...')
    state = {
        'config': args,
        'save_path': '',
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    
    if args.ema and 'ema_model' in kwargs.keys():
        state['ema_model'] = kwargs['ema_model']

    if save_cur:
        state['save_path'] = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth'))
        logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')))
    elif epoch % args.save_freq == 0:
        state['save_path'] = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth'))
        logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')))
    else:
        # state['save_path'] = 'current.pth'
        # torch.save(state, os.path.join(args.log_dir, 'current.pth'))
        print("not saving checkpoint")
        pass

LOADER_WK = None

def get_loader(args):
    # Init datasets and dataloaders
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet.scannet_detection_dataset import ScannetDetectionDataset
    from scannet.model_util_scannet import ScannetDatasetConfig

    DATASET_CONFIG = ScannetDatasetConfig()
    
    TEST_DATASET_SCANNET = ScannetDetectionDataset('val', num_points=args.num_point,
                                            augment=False,
                                            use_color=True if args.use_color else False,
                                            use_height=True if args.use_height else False,
                                            start_proportion=0.0,
                                            end_proportion=1.0)
    
    TEST_DATASET_ARKIT = ARKitSceneDataset('valid', num_points=args.num_point,
                                                augment=False,
                                                start_proportion=0.0,
                                                end_proportion=1.0,)

    test_loader_scannet = torch.utils.data.DataLoader(TEST_DATASET_SCANNET,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              worker_init_fn=my_worker_init_fn,
                                              pin_memory=True,
                                              drop_last=False)

    test_loader_arkit = torch.utils.data.DataLoader(TEST_DATASET_ARKIT,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               worker_init_fn=my_worker_init_fn,
                                               pin_memory=True,
                                               drop_last=False)
    
    if args.arkit:
        test_loader = test_loader_arkit
    else:
        test_loader = test_loader_scannet

    return test_loader, DATASET_CONFIG


LOADER_WK_ITER = None

def get_next_weak_batch():
    global LOADER_WK
    global LOADER_WK_ITER

    try:
        nxt = LOADER_WK_ITER.__next__()
    except:
        LOADER_WK_ITER = LOADER_WK.__iter__()
        nxt = LOADER_WK_ITER.__next__()
    
    return nxt


def get_model(args, DATASET_CONFIG, ema=False):
    if args.use_height:
        num_input_channel = int(args.use_color) * 3 + 1
    else:
        num_input_channel = int(args.use_color) * 3
    model = PQ_Transformer(num_class=DATASET_CONFIG.num_class,
                              num_heading_bin=DATASET_CONFIG.num_heading_bin,
                              num_size_cluster=DATASET_CONFIG.num_size_cluster,
                              mean_size_arr=DATASET_CONFIG.mean_size_arr,
                              input_feature_dim=num_input_channel,
                              num_proposal=args.num_target,
                              num_quad_proposal=args.quad_num_target,
                              sampling=args.sampling
                              )
    criterion = get_loss

    if ema:
        for param in model.parameters():
            param.detach_()
    return model, criterion

ema_model = None

def main(args):
    test_loader, DATASET_CONFIG = get_loader(args)
    model, criterion = get_model(args, DATASET_CONFIG)
    
    if args.ema:
        global ema_model
        ema_model, _ = get_model(args, DATASET_CONFIG, ema=True)
    
    if dist.get_rank() == 0:
        pass
        # logger.info(str(model))
    # optimizer
    if args.optimizer == 'adamW':
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "decoder" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "decoder" in n and p.requires_grad],
                "lr": args.decoder_learning_rate,
            },
        ]
        optimizer = optim.AdamW(param_dicts,
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    scheduler = get_scheduler(optimizer, 10000, args)
    model = model.cuda()
    if args.ema:
        ema_model = ema_model.cuda()
    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    if args.checkpoint_path:
        assert os.path.isfile(args.checkpoint_path)
        if args.ema:
            load_checkpoint(args, model, optimizer, scheduler, ema_model=ema_model)
        else:
            load_checkpoint(args, model, optimizer, scheduler, )

    # Used for AP calculation
    CONFIG_DICT = {'remove_empty_box': False, 'use_3d_nms': True,
                   'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
                   'per_class_proposal': True, 'conf_thresh': 0.0,'quad_thresh':0.5,
                   'dataset_config': DATASET_CONFIG, 'num_iou_quad': args.nms_iou_quad}

    # Directly Evaluate
    evaluate_one_epoch(test_loader, DATASET_CONFIG, CONFIG_DICT, args.ap_iou_thresholds, model, criterion, args)
    # evaluate_one_epoch(test_loader, DATASET_CONFIG, CONFIG_DICT, args.ap_iou_thresholds, ema_model, criterion, args, ema=True)
    
    return os.path.join(args.log_dir, f'ckpt_epoch_last.pth')

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def get_current_consistency_weight(epoch):
    global FLAGS
    args = FLAGS['args']
    def sigmoid_rampup(current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency_weight * sigmoid_rampup(epoch, args.consistency_rampup)


def evaluate_one_epoch(test_loader, DATASET_CONFIG, CONFIG_DICT, AP_IOU_THRESHOLDS, model, criterion, config, ema=False):
    
    stat_dict = {}
    start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    try:
        I = model.module.i
    except:
        I = model.i


    if config.num_decoder_layers > 0:
        prefixes = ['last_'] #, 'proposal_'] + [f'{i}head_' for i in range(config.num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
                          for iou_thresh in AP_IOU_THRESHOLDS]
    
    quad_ap_calculator_list = [QUADAPCalculator(iou_thresh, DATASET_CONFIG.class2quad, logger, I) \
                          for iou_thresh in AP_IOU_THRESHOLDS]
    

    mAPs = [[iou_thresh, {k: 0 for k in prefixes}] for iou_thresh in AP_IOU_THRESHOLDS]

    model.eval()  # set model to eval mode (for bn and dp)


    batch_pred_quad_map_cls_dict = {k: [] for k in prefixes}
    batch_gt_quad_map_cls_dict = {k: [] for k in prefixes}

    batch_pred_corner_dict = {k: [] for k in prefixes}
    batch_gt_corner_dict = {k: [] for k in prefixes}
    batch_gt_horizontal_dict = {k: [] for k in prefixes}

    for batch_idx, batch_data_label in enumerate(test_loader):
        for key in batch_data_label:
            if key == 'scan_name':
                continue
            
            batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)
        
        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = model(inputs)

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]

        for prefix in prefixes:


            batch_pred_quad_map_cls,pred_quad_mask,batch_pred_quad_corner = parse_quad_predictions(end_points, CONFIG_DICT, prefix)
            batch_gt_quad_map_cls,batch_gt_quad_corner = parse_quad_groundtruths(end_points, CONFIG_DICT)
            
            batch_pred_quad_map_cls_dict[prefix].append(batch_pred_quad_map_cls)
            batch_gt_quad_map_cls_dict[prefix].append(batch_gt_quad_map_cls)
            batch_pred_corner_dict[prefix].append(batch_pred_quad_corner)
            batch_gt_corner_dict[prefix].append(batch_gt_quad_corner)
            
            batch_gt_horizontal_dict[prefix].append(end_points['horizontal_quads']) 

            end_points['pred_quad_mask']=pred_quad_mask
            


        if (not config.arkit) and (batch_idx + 1) % config.print_freq == 0:
            logger.info(f'Eval: [{batch_idx + 1}/{len(test_loader)}]')
    
    #quad
    mAP_ = 0.0
    
    for prefix in prefixes:
        for (batch_pred_map_cls, batch_gt_map_cls,batch_pred_corner,batch_gt_corner,batch_gt_horizontal) in zip(batch_pred_quad_map_cls_dict[prefix],
                                                          batch_gt_quad_map_cls_dict[prefix],batch_pred_corner_dict[prefix],batch_gt_corner_dict[prefix],batch_gt_horizontal_dict[prefix]):
            for ap_calculator in quad_ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls,batch_pred_corner,batch_gt_corner,batch_gt_horizontal)
        # Evaluate average precision
        for i, ap_calculator in enumerate(quad_ap_calculator_list):
            metrics_dict = ap_calculator.compute_metrics()
            f1 = ap_calculator.compute_F1(calculated=True)
            logger.info(f'F1 scores: {f1}')
            # logger.info(f'=====================>{prefix} IOU THRESH: {AP_IOU_THRESHOLDS[i]}<=====================')
            # for key in metrics_dict:
            #     logger.info(f'{key} {metrics_dict[key]}')
            if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                mAP_ = metrics_dict['mAP']
            mAPs[i][1][prefix] = metrics_dict['mAP']
            ap_calculator.reset()

    for mAP_ in mAPs:
        logger.info(f'IoU[{mAP_[0]}]:\t' + ''.join([f'{key}: {mAP_[1][key]:.4f} \t' for key in sorted(mAP_[1].keys())]))
 
    return 0.0

if __name__ == '__main__':
    opt = parse_option()
    
    
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    initiate_environment(opt)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    import time
    LOG_DIR = os.path.join(opt.log_dir, 'pq-transformer',
                           f'{opt.dataset}_{RUN_NAME}')
    while os.path.exists(LOG_DIR):
        LOG_DIR = os.path.join(opt.log_dir, 'pq-transformer',
                               f'{opt.dataset}_{RUN_NAME}')
    opt.log_dir = LOG_DIR
    os.makedirs(opt.log_dir, exist_ok=True)

    logger = setup_logger(output=opt.log_dir, distributed_rank=dist.get_rank(), name="pq-transformer")
    if dist.get_rank() == 0:
        path = os.path.join(opt.log_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))
        logger.info(str(vars(opt)))

    ckpt_path = main(opt)
